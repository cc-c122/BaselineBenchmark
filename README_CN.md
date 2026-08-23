# BaselineBenchmark

面向多种硬件后端的算子 baseline 与 benchmark 工作区。

当前仓库只初始化项目结构和约定，不预置具体算子实现或 benchmark 逻辑。每个后端维护自己的算子实现和 benchmark 入口，跨后端复用的工具放在仓库公共目录中。

## 目录结构

```text
BaselineBenchmark/
├── backends/
│   ├── _template/
│   │   ├── ops/           # 新增后端时复制此模板
│   │   └── benchmarks/
│   ├── _aipu/
│   │   ├── ops/
│   │   └── benchmarks/
│   ├── ...
│   └── _nvidia/
│       ├── ops/
│       └── benchmarks/
├── common/                # 跨后端复用的 benchmark 工具和结果工具
├── configs/               # 共享的测试用例与 benchmark 配置
├── docs/                  # benchmark 约定和设计说明
├── results/               # 本地/生成的结果，只跟踪目录
└── scripts/               # 仓库级自动化和调度脚本
```

后端目录名称参考 `src/flaggems_vllm/runtime/backend`，保留前导下划线，便于与 runtime 中的后端一一对应。

## 新增后端

1. 将 `backends/_template` 复制为 `backends/<backend_name>`。
2. 将后端专属算子实现放入 `ops/`。
3. 将可执行 benchmark 脚本放入 `benchmarks/`。
4. 在 `configs/` 中新增或复用测试用例配置。
5. 在 benchmark 文档或结果元数据中记录命令、环境、硬件、软件版本和结果位置。

如果一个后端包含多个架构，可以继续增加一层目录，例如 `backends/_nvidia/ampere/ops/` 和 `backends/_nvidia/hopper/ops/`。

## Benchmark 约定

- 显式记录输入 shape、dtype、layout，以及 warmup 和测量次数；
- 在适用时提供 reference 或 baseline 实现；
- 将 correctness 检查与计时区间分开；
- 报告带单位的延迟统计和相关吞吐指标；
- 机器可读结果保存到 `results/`，不提交生成文件；
- 记录设备、驱动、框架、编译器和代码版本信息。

详细格式见 [`docs/benchmark_conventions.md`](docs/benchmark_conventions.md)。

## 当前状态

仓库当前仅包含项目骨架。后端可以独立添加实现，不需要改变顶层目录约定。

