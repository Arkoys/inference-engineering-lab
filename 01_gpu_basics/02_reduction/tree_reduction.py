"""
Tree reduction in pure Python.

This is a software-only preview of the GPU algorithm we'll eventually
write in CUDA. The idea is to make the reduction *shape* explicit:

    Round 0:   32 active elements  (stride = 1)   -- pair (0,1) (2,3) ...
    Round 1:   16 active elements  (stride = 2)
    Round 2:    8 active elements  (stride = 4)
    Round 3:    4 active elements  (stride = 8)
    Round 4:    2 active elements  (stride = 16)
    Round 5:    1 active element   (stride = 32 / final result)

Each round halves the number of "live" values by adding each element at
index i with the element at index i + stride. After 5 rounds, a
32-element array collapses to a single sum.

The reason this matters for GPUs:
    - 32 == warp size on NVIDIA, 32 == a typical thread block
    - "shared memory" in CUDA is the place where threads park these
      intermediate partial sums between rounds so they don't have to
      hit global memory over and over.
    - The same 5-round pattern shows up inside every fast CUDA reduction
      kernel (Mark Harris's classic "reduce" kernel, CUB, etc.)

So before we touch a GPU, we walk through the *algorithm* in plain
Python. The numbers we get will be terrible compared to torch.sum --
that's expected, because we're doing this with CPython interpreter
loops, not vectorized C or GPU code. The point is to see the structure.
"""

import time
import numpy as np
import torch


# ============================================================
# Algorithm
# ============================================================

def tree_sum(x):
    """Reduce a length-32 tensor to a scalar using a 5-round tree.

    Stride sequence: 1, 2, 4, 8, 16.
    After each round the number of "live" values halves:
        32 -> 16 -> 8 -> 4 -> 2 -> 1
    """
    assert x.numel() == 32, "tree_sum is hard-coded for a block of 32"

    # Work on a mutable copy so the caller's tensor isn't destroyed.
    buf = x.clone()

    # Round-by-round stride. Note: 5 strides collapse 32 -> 1.
    for stride in (1, 2, 4, 8, 16):
        # Each surviving index i adds the element at i + stride.
        # That is exactly half of the still-live elements, so this
        # is the "tree" shape: pairs at round 0, pairs-of-pairs at
        # round 1, etc.
        for i in range(0, 32, 2 * stride):
            buf[i] = buf[i] + buf[i + stride]

    return buf[0].item()


# ============================================================
# Sanity: same answer as torch.sum
# ============================================================

torch.manual_seed(0)
x = torch.rand(32, dtype=torch.float32)

mine = tree_sum(x)
ref = torch.sum(x).item()

print("Sanity check")
print("-" * 60)
print(f"  tree_sum : {mine:.10f}")
print(f"  torch.sum: {ref:.10f}")
print(f"  diff     : {abs(mine - ref):.2e}")
print()


# ============================================================
# Benchmark harness -- mirrors the style of reduction.py
# ============================================================

def bench(label, fn, iterations, warmup):
    for _ in range(warmup):
        fn()
    times = []
    for _ in range(iterations):
        start = time.perf_counter()
        fn()
        end = time.perf_counter()
        times.append(end - start)
    avg = np.mean(times) * 1000
    best = np.min(times) * 1000
    print(f"  {label:30s}  avg={avg:9.3f} ms   best={best:9.3f} ms")
    return avg, best


ITERATIONS = 100
WARMUP = 10

print("Benchmark: tree_sum vs torch.sum  (N = 32, one block)")
print("-" * 60)

t_avg, t_best = bench("tree_sum (Python loop)", lambda: tree_sum(x), ITERATIONS, WARMUP)
s_avg, s_best = bench("torch.sum(x).item()",     lambda: torch.sum(x).item(), ITERATIONS, WARMUP)

print()
print(f"  tree_sum vs torch.sum:  {t_avg / s_avg:8.1f}x slower (avg)")
print(f"                          {t_best / s_best:8.1f}x slower (best)")
print()
print("Reminder: this gap is not interesting on its own.")
print("The point is that the *algorithm shape* (5 rounds, stride doubling)")
print("is exactly what we will reuse inside a CUDA shared-memory block.")