# Chunk GLA 工作流实例索引

本目录是 operator_optimization_workflow.yaml 的已完成实例。

- 任务输入：operator_task.yaml
- 机器可读实验记录：experiment_record.yaml
- 完整人工复盘：
  /workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/doc/chunk_gla_optimization_notes.md
- Baseline 正确性：
  /workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/test/test_chunk_gla_reference.py
- Current 独立正确性：
  /workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/test/test_chunk_gla_current_reference.py
- 公平性能比较：
  /workspace/FlagGems-vllm/BaselineBenchmark/backends/_enflame/benchmarks/bench_chunk_gla.py

最终 baseline 是 Tops C++ 扩展调用 TopsAten，不是自定义设备 kernel。最终生产改动只保留
dense empty_like，代表 shape (8,1024,8,64) 从约 2.78 ms 降到 2.301 ms。
最终性能输出采用 baseline/current，大于 1 表示 current 更快。
