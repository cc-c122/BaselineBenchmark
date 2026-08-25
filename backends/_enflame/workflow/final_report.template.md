# <Operator> Enflame Baseline 与 Current 优化报告

## 1. 结论

- 最终 baseline：
- baseline 的真实实现与 lowering：
- 最终 current 改动：
- 正确性：
- 主要 shape 的 baseline/current：
- 已知限制：

## 2. 任务与环境

- 容器：
- 工作目录：
- 设备：
- dtype：
- 最终交付文件：
- baseline 文件：
- benchmark 文件：
- test 文件：

## 3. 数学定义和正确性红线

写出公式、state/layout/varlen/backward 语义，以及预注册容差。

## 4. Baseline 调查与选择

| 候选 | 实现方式 | 正确性 | 稳态性能 | 是否保留 | 原因 |
|---|---|---|---:|---|---|

## 5. 修改前 Current 基线

| Shape | Median | Min | Max | 首次编译 |
|---|---:|---:|---:|---:|

## 6. 实验记录

| ID | 假设 | 改动 | 正确性 | 性能变化 | 决策 |
|---|---|---|---|---:|---|

不得省略被拒绝实验。

## 7. 最终性能

比值定义：

```text
baseline/current = baseline_median_ms / current_median_ms
```

| Shape | Current ms | Baseline ms | Baseline/Current | 结论 |
|---|---:|---:|---:|---|

## 8. 正确性结果

列出 baseline 独立 oracle、current 独立 oracle、仓库原有 forward/backward 测试，
以及所有 xfail 和数值边界。

## 9. 测试与复现命令

提供从进入容器、cd 到每条 pytest/benchmark 的完整命令。

## 10. 最终 Diff 与目录

只列出最终保留的文件和改动，不混入已撤销实验。

## 11. 剩余瓶颈与下一步

区分：

- 可继续的局部优化；
- 需要 kernel/算法结构性重写的方向；
- 厂商 API 或硬件能力限制。
