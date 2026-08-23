# Backends

Each backend directory is self-contained:

```text
backends/<backend_name>/
├── ops/                    # Backend-specific operator implementations
└── benchmarks/             # Baseline and benchmark entry points
```

Keep backend-only dependencies and device-specific setup inside the backend
directory whenever practical. Shared code belongs in `common/` and should not
silently depend on one particular backend.

The initial backend directories correspond to the backends currently present
under `flaggems_vllm/runtime/backend`. `_template` is not a runnable backend;
it is the starting point for future additions.

