# Enflame Chunk GLA 优化记录

更新日期：2026-08-25

## 1. 范围与环境

本文记录 Chunk GLA 在燧原 GCU 上的优化经验、失败实验、瓶颈、测试方法和后续方向。

相关实现：

- 最终适配：`/workspace/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_enflame/ops/chunk_gla.py`
- Tops C++ baseline：`/workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/ops/chunk_gla/chunk_gla_tops.py`
- C++ 热路径：`/workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/ops/chunk_gla/chunk_gla_tops_extension.cpp`

所有实验只在 Docker 容器 `czy-enflame3` 内完成。设备是 GCU major 3、minor 0，24 SIP，主要测试 BF16 输入。

## 2. 数学定义与正确性红线

独立数学定义采用逐 token 状态递推：

```text
h_t = exp(g_t) * h_(t-1) + k_t outer v_t
o_t = scale * q_t @ h_t
```

`g` 是非正的 log forget gate。任何性能优化必须满足：

1. 不能通过放宽容差掩盖公式、stride、mask 或状态布局错误。
2. 输出和最终状态必须全部有限。
3. 强负 gate 不能产生 NaN/Inf。
4. 覆盖 initial state、final state、`state_v_first` 和 varlen。
5. 使用独立 CPU FP64 逐 token oracle，不能只把 Triton 当参考答案。
6. 保持 causal 语义和 chunk 边界状态递推完全一致。

固定大 chunk 会让中点指数分解先产生 Inf 和 0，最终形成 NaN。baseline 因此使用数据相关的安全 chunk：普通 logsigmoid 输入通常取 128，`g=-4` 压力输入降到 32。此前固定 256 得到的更快数字不具备有效正确性，不能使用。

## 3. 当前 Triton 实现结构

生产前向分为：

```text
chunk_local_cumsum
    -> chunk_fwd_h
    -> chunk_gla_fwd_intra_gk
    -> chunk_gla_fwd_o_gk
```

已存在的 Enflame 专项优化：

- BF16 矩阵输入、FP32 accumulator。
- gate 缩放在 tile 内完成。
- 因果 mask 在寄存器/tile 内处理。
- 压缩严格下三角 block，避免调度必然为空的 block。
- K<=256 使用直接 A kernel，宽 K 使用 BK=128 分裂并归并。
- 三维 grid 重映射。
- TLE 异步加载输出 kernel。
- 一个 program 合并多个 value tile，复用 q/g/A。
- dense BF16 并行 local-state 和 state scan。
- backward 直接写目标 dtype，减少全张量转换。
- chunk 在短序列上选 16/32，最大为 64。

## 4. 最终 baseline 使用什么实现

最终 baseline 不是 Triton，也不是 Torch eager DSL，而是：

```text
Python 参数检查和 JIT 构建
    -> PyTorch C++ Extension
    -> Tops C++ / TopsAten C++ API
       - topsatenCumsum
       - topsatenExp
       - topsatenBmm
```

Python 不承担热路径数学计算。

### 4.1 保留的优化

原始 Tops C++ 每个 chunk 单独执行 cumsum 和五组 exp。最终采用维度自适应：

- `K <= 64`：一次预计算全部 chunk 的 cumsum 和门控 factor，把随 chunk 数增长的 cumsum/exp launch 降为固定次数。
- `K > 64`：保留逐 chunk 低内存路径，避免宽 K 下物化巨大的 FP32 factor。

结果：

- K=64 从旧版约 5.24 ms 降到约 4.71 ms，提升约 10%。
- K=128 两种路径基本持平。
- K=512 批量路径略慢，所以选择逐 chunk。

## 5. 最终性能

同一 benchmark 的中位数：

| B | T | H | K/V | 当前 Triton | Tops C++ baseline | baseline/current |
|---:|---:|---:|---:|---:|---:|---:|
| 8 | 1024 | 8 | 64 | 2.565 ms | 4.709 ms | 1.84x |
| 4 | 2048 | 16 | 128 | 6.225 ms | 15.808 ms | 2.54x |
| 4 | 1024 | 8 | 512 | 8.293 ms | 18.533 ms | 2.23x |

比值大于 1 表示当前 Triton 更快。当前实现约快 1.84x 到 2.54x。

## 6. 已验证但未保留的方案

### 6.1 Torch eager baseline

Torch 版本经过优化和同口径比较后，Tops C++ 在代表性形状上稳定快约 1% 到 4%，因此删除 Torch GLA 候选。MLA 等无关实现保留。

### 6.2 TCLE 自定义 `__global__` kernel

以下链路已经实际跑通：

```text
topscc -arch gcu300
    -> TCLE __global__ kernel
    -> PyTorch 当前 GCU stream
```

说明环境允许编写真正运行在 GCU 上的设备 C++ kernel。但未保留，原因是：

1. GLA 热点是 QK^T、A@V、Q@state、K^T@V。
2. SDK 的 `tcle::matrix_t` 官方样例只支持 gcu400；当前设备为 GCU300。
3. GCU300 公开 TCLE 无法表达与 Triton `tl.dot` 对等的矩阵 tile。
4. 设备 kernel 内不能调用 host 侧 `topsatenBmm`。
5. 修复正确性后，K=64 约 45.5 ms，K=512 约 184 ms，远慢于 TopsAten。

该实验曾因 q/k slice 与 cumsum 输出 stride 不同，出现约一半元素错误和尾部 NaN。强制紧凑化后通过测试，但性能不合格，相关候选已删除。

### 6.3 TopsAten 混合输出 dtype

实测 TopsAten BMM 要求：

```text
lhs.dtype == rhs.dtype == output.dtype
```

BF16 lhs/rhs、FP32 output 返回 `TOPSATEN_STATUS_BAD_PARAM`。因此 baseline 无法直接表达 Triton 常用的 BF16 matrix input + FP32 output 数据流，只能使用 FP32 factor/BMM 来守住递推精度，代价是转换、带宽和吞吐。

### 6.4 固定 chunk_size=256

普通长序列和强负 gate 均出现过非有限 factor，因此否决。性能数据必须包含安全 chunk 选择开销。

## 7. 主要不足与瓶颈

### 7.1 Tops C++ baseline

- C++ 是 host 调度代码，不会把整个 GLA 自动 lowering 成一个设备 kernel。
- 每个 TopsAten 调用之间存在 kernel launch 和显存边界。
- q/k/v/g 需要 transpose、contiguous 和 FP32 转换。
- FP32 BMM 不具备 BF16 input + FP32 accumulator 的吞吐优势。
- scores、factor 和 state update 物化到全局显存。
- chunk state 依赖让部分 BMM 串行。
- 公开 GCU300 TCLE 缺少矩阵 tile intrinsic。

### 7.2 当前 Triton 实现

- 完整 FP32 `g_cumsum[B,T,H,K]` 会写回并被多个阶段读取。
- FP32 `A[B,T,H,BT]` 单独物化，然后由输出 kernel 读取。
- 并行 state 先物化 FP32 `local_state[B,NT,H,K,V]`，再 scan 并写 h。
- dense 输出也使用 `torch.zeros_like(v)`，存在一次预清零。
- varlen、`state_v_first` 和部分 dtype/layout 不满足并行 state specialization，会退回慢路径。
- TLE/autotune 配置空间会增加首次编译时间。

## 8. 当前生产实现还有哪些优化空间

有，但当前 dense BF16 主路径已经较成熟，应按优先级推进。

### P0：优先验证

1. **varlen segmented parallel state scan**

   当前并行 state 要求 `cu_seqlens is None`。可以基于 chunk_indices/split_offsets 实现 segmented local-state + scan，避免 varlen 退回慢路径。

2. **覆盖 state_v_first 的并行 state**

   当前 `state_v_first=True` 退出并行 specialization。可以实现 V-first tile，或在片上做布局转换。

3. **V<=64 的 A+output 融合**

   当前先写 FP32 A，再读取做 A@V。V 只有一个 tile 时，可尝试融合，省去 A 的写回和读取。V 较大时 A 会被多个 value tile 复用，不应直接融合。

4. **dense 输出使用 empty**

   当 `cu_seqlens is None` 且 kernel 确认覆盖全部 B/T/H/V 时，评估 `torch.empty_like(v)`，省去清零；varlen/padding 必须保留 zero 语义。

### P1：需要系统 benchmark

1. 按 shape/dtype 搜索 chunk 32/64，谨慎探索 128；必须继续检查指数稳定性。
2. 用分组或层次化 prefix scan 减少 local_state 与 h 的显存往返。
3. 对常用 vLLM shape 离线固化 BK/BV/B_VCHUNK/warp/stage 配置，降低首次编译成本。
4. 扩充有限的 TLE value grouping 候选；不能盲目增大 accumulator 数量。
5. 对 FP16 研究动态安全中心，使部分形状从 BC16 提升到 BC32/64。

### P2：训练场景

- 调优 backward state scan 和 dq/dk/dv/dg tile。
- 若交付目标是 vLLM 推理，应优先 dense BF16 forward。

## 9. 消融实验

形状 `(8,1024,8,64)`：

- 默认当前实现约 2.79 ms。
- 关闭 `FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE` 后约 189.09 ms，慢约 68x。

并行 state 是当前最关键的优化，后续不能破坏该 specialization。

关闭 `FLA_GLA_TLE` 会进入另一套大规模 autotune/编译路径，超过 30 秒仍未完成。该结果只能说明 TLE 路径不能轻易移除，不记录不可靠的稳态数字。

## 10. 测试和复现

正确性测试：

```text
/workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/test/test_chunk_gla_reference.py
```

```bash
cd /workspace/FlagGems-vllm
python -m pytest -q BaselineBenchmark/backends/_enflame/test
```

当前结果：

```text
8 passed, 1 warning
```

唯一 warning 是 GCU 将 Long 索引替换为 Int。测试覆盖普通 gate、强负 gate、finite 检查、initial/final state、两种 state layout、varlen、K=128 宽维度分支、正 gate 拒绝和 CPU FP64 oracle。

性能测试：

```text
/workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/benchmarks/bench_chunk_gla.py
```

```bash
cd /workspace/FlagGems-vllm
python BaselineBenchmark/backends/_enflame/benchmarks/bench_chunk_gla.py \
  --warmup 2 --iter 10
```

输出中的 `baseline=tops_cpp_topsaten` 表示最终 baseline 类型。

## 11. 后续优化验收标准

1. 从干净进程运行，避免多个 autotune/编译任务并发污染。
2. warmup 后同步计时，报告中位数并保留 min/max。
3. 至少覆盖 K/V=64、128、512 和长序列。
4. 同时测试 finite、FP64 oracle、initial/final state 和 varlen。
5. 分别记录首次编译时间和稳态执行时间。
6. 未稳定超过现实现的候选必须删除，不能因为实现更底层而保留。


## 12. State 的数学含义与开关语义

GLA 的 state 是按时间递推的矩阵记忆。对每个 batch 和 head，K-first 布局下
`h_t` 的形状为 `[K,V]`，整体形状为 `[B,H,K,V]`：

```text
h_t = diag(exp(g_t)) * h_(t-1) + k_t outer v_t
o_t = scale * q_t^T * h_t
```

其中 `exp(g_t)` 是逐 K 维的遗忘系数，`k_t outer v_t` 把当前 token 写入记忆，
`q_t` 从记忆中读取输出。设置 `state_v_first=True` 只把存储布局改成
`[B,H,V,K]`，数学定义不变。

需要区分三类“开关”：

1. 数学上的 state 没有关闭模式。GLA 即使不返回 state，也必须在内部递推 state。
2. `initial_state=None` 表示从全零状态开始，不表示关闭 state。
3. `output_final_state=False` 只是不把最后的 `h_T` 返回给调用者，内部仍计算各 chunk
   边界 state。
4. `FLAGGEMS_ENFLAME_GLA_PARALLEL_STATE` 是实现算法开关，不是模型语义开关。
   当前默认开启，并行生成局部 state 后做 chunk scan。关闭后会退回串行实现；
   在 `(8,1024,8,64)` 上约从 2.79 ms 退化到 189.09 ms，因此正常使用必须保持开启。

所以“正常 GLA 的 state 是打开还是关闭”的准确答案是：数学上始终存在；通常
`initial_state=None`、`output_final_state=False`，即零初态且不返回终态；Enflame
实现的并行 state 优化默认开启。

## 13. 生产 Triton GLA 第二轮优化实验（2026-08-25）

目标文件：

```text
/workspace/FlagGems-vllm/src/flaggems_vllm/runtime/backend/_enflame/ops/chunk_gla.py
```

### 13.1 接受的优化：dense 输出免清零

原实现无条件执行：

```python
o = torch.zeros_like(v)
```

逐项检查普通和 TLE output kernel 后确认：当 `cu_seqlens is None` 时，grid 对所有
`B/T/H/V` 有效元素形成完整且唯一的覆盖写，初值不会参与计算。因此改为：

```python
o = torch.empty_like(v) if cu_seqlens is None else torch.zeros_like(v)
```

varlen 路径仍保留清零，因为 vLLM 可能读取 padding/gap，不能改变其零语义。

严格 A/B 形状 `(8,1024,8,64)`，BF16，`warmup=5, iter=30`：

| 版本 | 三轮 current 中位数 |
|---|---|
| 修改前 | 2.785 / 2.782 / 2.779 ms |
| dense empty | 2.534 / 2.302 / 2.299 ms |
| 最终复测 | 2.301 ms |

稳定轮次约从 2.78 ms 降到 2.30 ms，提升约 17%。最终同轮 Tops C++ baseline 为
4.687 ms，baseline/current=2.04x，即当前 Triton 实现约快 2.04 倍。

另外两个代表形状的复测：

| shape | 修改前参考 | 修改后 |
|---|---:|---:|
| (4,2048,16,128) | 6.225 ms | 6.090 ms |
| (4,1024,8,512) | 8.293 ms | 8.184 ms |

这两个形状收益较小，约 1%–2%，符合输出清零在总耗时中占比随矩阵计算量增大而下降的预期。

### 13.2 否决并撤销的方向

1. **chunk size 32/128**

   当前并行 state specialization 明确要求 `BT=64`。改成 32 或 128 会进入串行 state
   路径；结合关闭并行 state 的 189.09 ms 消融结果，没有继续大规模 autotune 的价值。
   实验开关已删除，最终仍固定最大 chunk 64。

2. **把前向 A 从 FP32 改为 BF16 存储**

   理论上可把 A 的显存流量减半，且前向 `A@V` 前本来会转为输入 dtype；但实测稳定
   轮次为 2.672/2.675 ms，慢于 FP32 A 的约 2.30 ms。同时 backward dv 明确依赖 FP32 A
   的精度。该改动已撤销。

3. **BF16 固定 BC16/BC32 保守分块**

   独立 FP64 测试发现连续 `g=-4` 时，BC64 对角块的指数中点分解会形成 `inf*0`，
   尽管最终数学结果有限。固定 BC16 能通过该测试，但正常形状为约 7.25–7.30 ms；
   固定 BC32 约 3.586 ms，均有明显性能退化，已撤销。

4. **host 端动态 gate span 选择 BC**

   数学上可以按累计 gate span 选择 64/32/16，但 `.item()` 会在每次调用中引入
   CPU/GCU 同步。实测无法进入可接受的 benchmark 周期，已撤销。

5. **设备端安全标志 + 极端 chunk 稳定覆盖**

   实现过一个不做 host 同步的原型：正常 chunk 走 BC64，设备端标记危险 chunk，
   仅危险 chunk 用直接 `g_i-g_j <= 0` 公式覆盖。连续 `g=-4` 的 FP64 测试通过，
   但正常形状因额外 kernel 和大量早退 program 退化到 8.843 ms，且首次编译成本高，
   因此完整撤销。

6. **关闭 TLE output 路径**

   会进入另一套大规模 autotune/编译路径，不能得到有意义的短时稳态数据；结合已有
   TLE 优化结果，不继续。

最终生产文件相对实验前只有 dense `empty_like` 一处功能改动，没有保留实验开关、
安全标志 kernel 或 BF16 A。

### 13.3 新增独立正确性测试

新增：

```text
/workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/test/test_chunk_gla_current_reference.py
```

它使用逐 token CPU FP64 定义式，不依赖另一份 GLA 实现，覆盖：

- 普通 dense BF16；
- initial/final state；
- K-first 和 V-first state；
- varlen 独立序列；
- K=128 宽 key；
- 连续强负 gate 的数值边界。

结果：

```text
5 passed, 1 xfailed
```

唯一 strict xfail 是连续 `g=-4` 的既有 BC64 中点指数溢出问题。该项故意保留，
用于防止未来把已知数值边界误报为通过。

现有实现互证的 BF16 前向+反向测试：

```bash
pytest -q tests/test_FLA/test_chunk_gla.py -k "B1-T256-H4-D32" -s
```

结果 `1 passed, 13 deselected`。输出、final state、dq/dk/dv/dg/dh0 的相对误差均
低于原测试阈值，其中最大报告 ratio 为 dh0 的 0.002341。

### 13.4 当前停止继续修改的理由

已穷尽本轮低风险局部方向。剩余可能产生数量级收益的工作都属于结构性重写：

1. 把 A 生成和 `A@V` 在 V 较小时融合，避免 FP32 A 落显存；
2. varlen segmented parallel state scan；
3. `state_v_first` 专用并行 state；
4. 把 gate 安全检测融合进已有 A kernel，避免额外 launch，并提供真正稳定的危险块算法；
5. 对 state scan 做分组/层次化处理，减少 `local_state -> h` 显存往返。

这些方向需要新 kernel、完整多 shape 调优和长时间编译验证，已不再是可安全接受的局部
优化。继续在现有 Python 调度层叠加分支，实验已证明会抵消收益。


## 14. 可复用 AI 算子优化工作流

本次过程已经抽象为可复用工作流：

```text
/workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/workflow/
```

入口说明：

```text
workflow/README.md
```

机器可读主流程：

```text
workflow/operator_optimization_workflow.yaml
```

新算子任务输入模板、实验模板和验收清单：

```text
workflow/operator_task.template.yaml
workflow/experiment_record.template.yaml
workflow/final_report.template.md
workflow/ACCEPTANCE_CHECKLIST.md
```

本次 chunk GLA 的机器可读示例：

```text
workflow/runs/chunk_gla/operator_task.yaml
workflow/runs/chunk_gla/experiment_record.yaml
workflow/runs/chunk_gla/final_report.md
```

后续选择新算子时，先复制 task 模板填写目标、交付文件、数学语义、shape、dtype 和必需模式，
再要求 AI 严格执行主流程 YAML。AI 必须自主调查 baseline、建立独立 oracle、优化正确
baseline、优化 current、逐项回滚失败实验，并按验收清单输出最终结果。
