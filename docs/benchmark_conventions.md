# Benchmark conventions

This document defines the minimum metadata expected from a benchmark. It is a
starting point and can be extended as more backends are added.

## Required case information

Every benchmark case should identify:

- operator name and implementation under test;
- baseline/reference implementation;
- input shapes, dtypes, layouts, and relevant flags;
- device and backend;
- warmup iterations and measured iterations;
- correctness tolerance and comparison method.

## Timing

Correctness checks must run before timing. The timed region should contain only
the operation being measured. Use the device's appropriate synchronization
mechanism before reading timestamps, and state whether setup, memory
allocation, compilation, and synchronization are included.

At minimum, report:

- number of samples or iterations;
- mean latency;
- median latency;
- p99 latency when enough samples are available;
- throughput with its unit when meaningful.

## Result metadata

Prefer one machine-readable result file per run. A result should include the
case parameters plus environment metadata such as:

```yaml
backend: _example
device: example-device
operator: example_op
dtype: float16
shape: [1024, 4096]
warmup: 10
iterations: 100
latency_us: 0.0
framework_version: unknown
driver_version: unknown
commit: unknown
```

Generated result files belong under `results/` and should remain untracked.

