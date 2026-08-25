# 算子优化最终验收清单

## 范围与环境

- [ ] 只在指定 Docker 容器内修改。
- [ ] 已记录设备、镜像、依赖、环境变量和 git status。
- [ ] 未覆盖任何无关用户改动。
- [ ] 没有并发 benchmark、pytest 或 compiler 污染计时。

## 数学与正确性

- [ ] 已写出独立数学定义。
- [ ] oracle 不调用 baseline/current 核心实现。
- [ ] 容差在看到实验结果前定义，并符合 dtype。
- [ ] 普通、tail、非二次幂和极端数值已覆盖。
- [ ] state/layout/varlen/backward 的任务必需组合已覆盖。
- [ ] 所有输出和 state 均检查 finite。
- [ ] xfail 和已知限制被明确披露，没有计为通过。

## Baseline

- [ ] 调查过官方、仓库、Torch、Tops C++ 和可用 DSL。
- [ ] 没有官方 baseline 时至少比较过两个合理方向，或说明不可行证据。
- [ ] 最终 baseline 通过独立 oracle。
- [ ] 最终 baseline 是正确候选中最快的。
- [ ] 报告准确说明其最终 lower 到 TopsAten、Triton 或设备 kernel。

## Benchmark

- [ ] baseline/current 使用相同输入、dtype、shape、布局和语义。
- [ ] 计时前后均同步 GCU。
- [ ] 报告 median/min/max，且 warmup 与 iter 可配置。
- [ ] 首次编译和稳态时间分开。
- [ ] 至少三轮独立运行，或使用随机交错 A/B。
- [ ] 最后一列为 baseline/current，大于 1 表示 current 更快。
- [ ] 性能变化超过修改前噪声区间。

## Current 优化

- [ ] 每项实验只有一个主要假设。
- [ ] 每项实验有修改前备份和回滚方式。
- [ ] 每项实验先正确性、后性能。
- [ ] 被拒绝实验已撤销并记录原因。
- [ ] 最终代码没有临时环境变量、调试输出或无用 kernel。
- [ ] 最终 diff 只包含通过验收的改动。

## 交付

- [ ] baseline 源码位于 `ops/<operator_name>/`。
- [ ] 性能脚本位于 `benchmarks/`。
- [ ] 正确性脚本位于 `test/`。
- [ ] 优化过程位于 `doc/`。
- [ ] 给出所有可直接复制的测试与 benchmark 命令。
- [ ] 给出最终性能表、接受/拒绝实验和剩余瓶颈。
