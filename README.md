# BaselineBenchmark

Baseline and benchmark workspace for operator libraries running on different
hardware backends.

This repository is intentionally implementation-free at initialization. Each
backend owns its operator implementations and benchmark entry points, while
shared utilities and conventions live at the repository level.

## Directory layout

```text
BaselineBenchmark/
├── backends/
│   ├── _template/
│   │   ├── ops/           # Copy this layout when adding a backend
│   │   └── benchmarks/
│   ├── _aipu/
│   │   ├── ops/
│   │   └── benchmarks/
│   ├── ...
│   └── _nvidia/
│       ├── ops/
│       └── benchmarks/
├── common/                # Shared benchmark helpers and result utilities
├── configs/               # Shared case definitions and benchmark settings
├── docs/                  # Benchmark conventions and design notes
├── results/               # Local/generated results; only the directory is tracked
└── scripts/               # Repository-level automation and orchestration
```

The backend names mirror `src/flaggems_vllm/runtime/backend`. The leading
underscore is retained so a backend can be mapped back to the runtime backend
without ambiguity.

## Adding a backend

1. Copy `backends/_template` to `backends/<backend_name>`.
2. Put backend-specific implementations under `ops/`.
3. Put executable benchmark programs under `benchmarks/`.
4. Add or reuse case definitions in `configs/`.
5. Record the command, environment, hardware, software versions, and result
   location in the benchmark's documentation or result metadata.

For a backend with architecture-specific variants, use another level below
the backend, for example `backends/_nvidia/ampere/ops/` and
`backends/_nvidia/hopper/ops/`.

## Benchmark conventions

Benchmarks should make correctness and performance independently verifiable:

- keep input shapes, dtypes, layouts, and warmup/measurement counts explicit;
- include a reference or baseline implementation when applicable;
- separate correctness checks from timed regions;
- report latency statistics and relevant throughput metrics with units;
- save machine-readable results under `results/` rather than committing
  generated output;
- document device, driver, framework, compiler, and commit information.

See [`docs/benchmark_conventions.md`](docs/benchmark_conventions.md) for the
initial format guidance.

## Running benchmarks

There is no mandatory runner or dependency lockfile yet. Backend benchmarks
may be run directly from their own directory while the common runner is being
developed. A benchmark should document its command in its module-level help or
an adjacent README.

Typical usage after adding a benchmark is:

```bash
cd /path/to/BaselineBenchmark
python backends/<backend_name>/benchmarks/<benchmark>.py --help
```

## Status

The repository currently contains only the project skeleton. Backend owners
can add implementations independently without changing the top-level layout.

