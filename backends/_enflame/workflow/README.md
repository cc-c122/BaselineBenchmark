# Enflame 算子 Baseline 与优化工作流

这套工作流用于把“给定一个算子和目标”转化为 AI 可自主执行、可审计、可复现的优化任务。

## 1. 输入

先复制并填写：

```text
workflow/operator_task.template.yaml
```

最少需要指定：

- 算子名称和最终交付文件；
- 功能、输入输出、数学定义或参考实现；
- 目标设备、容器和工作目录；
- dtype、shape、布局、state、varlen 等语义；
- 优化目标：吞吐、延迟、显存或编译时间；
- 必须保持的正确性边界。

如果没有官方 baseline，AI 必须依次调查官方实现、仓库实现、Torch、Tops C++/TopsAten
和可用 DSL，并实现至少两个合理候选；只保留“通过独立正确性校验后最快”的一个。

## 2. 目录约定

每个算子按名字独立存放：

```text
BaselineBenchmark/backends/_enflame/
├── ops/<operator_name>/                 # baseline 源码和构建产物
├── benchmarks/bench_<operator_name>.py  # baseline/current 公平性能比较
├── test/test_<operator_name>_reference.py
├── test/test_<operator_name>_current_reference.py
├── doc/<operator_name>_optimization_notes.md
└── workflow/
```

不要把 baseline 写进最终交付算子，也不要让正确性 oracle 复用 current 的核心公式或 kernel。

## 3. AI 执行入口

把任务 YAML 和下面的要求交给 AI：

```text
严格执行
/workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/workflow/operator_optimization_workflow.yaml。
读取 operator_task.yaml，先建立独立正确性 oracle 和修改前性能基线，再寻找并优化 baseline，
随后优化 current op。每个候选都写入 experiment_record.yaml；正确性失败或收益不稳定必须撤销。
只在指定 Docker 容器内操作，最终给出测试命令、性能表、接受/拒绝实验和剩余瓶颈。
```

机器可读主流程：

```text
workflow/operator_optimization_workflow.yaml
```

实验记录模板：

```text
workflow/experiment_record.template.yaml
```

最终报告模板：

```text
workflow/final_report.template.md
```

人工/AI 双重验收清单：

```text
workflow/ACCEPTANCE_CHECKLIST.md
```

## 4. 核心原则

1. 数学正确性是红线，性能不能交换语义。
2. baseline 必须独立、正确、可解释，不以“比 current 慢”为目标。
3. baseline 和 current 使用相同输入、dtype、shape、语义和同步计时方法。
4. 首次编译时间与稳态执行时间分开报告。
5. 一次只验证一个主要假设，保留修改前数据和精确差异。
6. 小于噪声区间的变化不算收益；至少三轮独立进程或同进程随机交错 A/B。
7. 失败实验也必须记录，并撤销代码。
8. 结束时生产代码中只保留正确且稳定获益的改动。

## 5. 本次 GLA 可复用经验

- state 是算法语义，`output_final_state=False` 只是不返回终态。
- 不能直接把 NVIDIA 的 tile/chunk 结论当成 GCU 最优值。
- 改 chunk 大小时必须保证 state、A、output、backward 都获得同等级 specialization。
- `torch.zeros_like` 到 `empty_like` 只有在证明 kernel 完整覆盖写时才安全。
- Tops C++ Python 封装不等于自定义设备 kernel；应明确最终落到 TopsAten 还是设备代码。
- 多个编译/autotune 任务并发会污染数据；每次实验前检查残留进程。
- 极端 gate、tail、varlen、state layout 和宽维度必须独立测试。
- 无同步的 device fallback 也可能因额外 launch 和早退 program 大幅变慢。
