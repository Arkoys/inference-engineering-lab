"""
Sanity check: torch.sum(A) vs A.sum() on CPU.

These should hit the exact same code path. If timings diverge, something
is fishy. Also: add an explicit torch.cpu.synchronize() to verify timing
isn't being affected by async behavior we forgot about.
"""

import numpy as np
import torch

sizes = [
    100_000,
    1_000_000,
    10_000_000,
]

ITERATIONS = 100
WARMUP = 10


def bench(label, fn):
    times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        s = fn()
        # explicit CPU sync — on CPU this is a no-op but proves the call
        # is already synchronous (if timings change, we had a bug)
        torch.cpu.synchronize()
        end = time.perf_counter()
        times.append(end - start)
    avg = np.mean(times) * 1000
    best = np.min(times) * 1000
    print(f"  {label:30s}  avg={avg:8.3f} ms   best={best:8.3f} ms")
    return avg, best


import time  # placed here only so the function above reads top-to-bottom

for N in sizes:
    print("=" * 60)
    print(f"N = {N:,}")
    print("-" * 60)

    A_cpu = torch.rand(N, dtype=torch.float32, device="cpu")

    # Warm-up for BOTH variants so neither is penalized for first-time cost
    for _ in range(WARMUP):
        torch.sum(A_cpu)
        A_cpu.sum()

    print("Experiment A: torch.sum(A)")
    a_avg, a_best = bench("torch.sum(A)", lambda: torch.sum(A_cpu))

    print("Experiment B: A.sum()")
    b_avg, b_best = bench("A.sum()",       lambda: A_cpu.sum())

    print()
    print(f"  A vs B ratio (avg):  {a_avg / b_avg:.3f}x")
    print(f"  A vs B ratio (best): {a_best / b_best:.3f}x")
    print()