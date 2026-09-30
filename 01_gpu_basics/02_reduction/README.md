# Reduction Benchmark

A reduction takes an array and collapses it to a single value.
`torch.sum(A)` is the canonical example: sum all elements into one scalar.

This directory contains a four-level performance comparison to make
one idea concrete:

```
Python loop
    ↓
naive CPU algorithm
    ↓
optimized PyTorch CPU
    ↓
optimized GPU reduction
```

The same mathematical operation — sum N float32 numbers — is implemented
at four different levels. Each level is dramatically faster than the
one above it, and **most of that speedup comes from the software, not
the hardware**.

## Files

- `reduction.py` — benchmarks naive Python vs `torch.sum` (CPU) vs MPS
- `sanity_check.py` — earlier check that `torch.sum(A)` and `A.sum()`
  route to the same kernel

## Why this matters

A custom CUDA kernel doesn't beat `torch.sum` because you bought a
faster GPU. It beats `torch.sum` because you wrote code that exploits
specific hardware features (vector loads, shared memory, warp shuffles,
hierarchical tree reduction) that the general-purpose kernel can't
fully use.

Before optimizing, we establish the baseline: **how fast is each
abstraction layer, and where does the time actually go?**
