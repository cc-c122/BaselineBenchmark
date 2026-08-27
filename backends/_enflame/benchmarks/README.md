# Enflame S60 sparse-attention benchmarks

This directory contains controlled same-card benchmarks for:

- NSA explicit block-index forward;
- MSA BF16 prefill;
- MSA BF16 decode.

Each process measures exactly one implementation. Run the Torch
reference and adapted implementation in separate processes with the
same arguments.

Required environment:

```bash
export PYTHONPATH=/path/to/BaselineBenchmark:/path/to/FlagAttention/src
export ENFLAME_PT_OP_DEBUG_CONFIG='op_sync_mode=false op_statistics=false'
```

Examples:

```bash
python backends/_enflame/benchmarks/bench_nsa_forward.py           --implementation torch --case CALIB_H16_S2K --device 1

python backends/_enflame/benchmarks/bench_nsa_forward.py           --implementation adapted --case CALIB_H16_S2K --device 1

python backends/_enflame/benchmarks/bench_msa_prefill.py           --implementation torch --case P2 --device 1

python backends/_enflame/benchmarks/bench_msa_prefill.py           --implementation adapted --case P2 --device 1

python backends/_enflame/benchmarks/bench_msa_decode.py           --implementation torch --case D4 --device 1

python backends/_enflame/benchmarks/bench_msa_decode.py           --implementation adapted --case D4 --device 1
```

Timing protocol:

- explicit warmup;
- sustained inner-loop execution;
- synchronization outside each measured group;
- median/min/max latency in milliseconds per invocation;
- optional JSON result output;
- correctness checks are kept outside timed benchmarks and live under
  `backends/_enflame/test/`.
