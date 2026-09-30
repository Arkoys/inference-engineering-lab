import time
import numpy as np
import torch


# ============================================================
# Implementations
# ============================================================

def naive_sum(A):
    """The most primitive Python loop possible.

    No NumPy. No vectorization. No C code.
    Pure CPython interpreter overhead per element.
    """
    total = 0.0
    for x in A:
        total += x
    return total


def torch_sum(A):
    """Optimized PyTorch reduction. Runs in C, vectorized with SIMD."""
    return A.sum().item()


# ============================================================
# Benchmark harness
# ============================================================

sizes = [10_000, 100_000, 1_000_000]
ITERATIONS = 20
WARMUP = 5


for N in sizes:
    print("=" * 60)
    print(f"N = {N:,}")
    print()

    A_cpu = torch.rand(N, dtype=torch.float32, device="cpu")

    # ----- naive Python -----------------------------------------
    # No warm-up for naive: it would already dominate total runtime.
    naive_times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        _ = naive_sum(A_cpu)
        end = time.perf_counter()
        naive_times.append(end - start)
    naive_avg = np.mean(naive_times) * 1000
    naive_min = np.min(naive_times) * 1000
    print(f"  naive Python    avg={naive_avg:10.3f} ms   best={naive_min:10.3f} ms")

    # ----- torch CPU --------------------------------------------
    for _ in range(WARMUP):
        torch_sum(A_cpu)
    torch_cpu_times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        _ = torch_sum(A_cpu)
        end = time.perf_counter()
        torch_cpu_times.append(end - start)
    torch_cpu_avg = np.mean(torch_cpu_times) * 1000
    torch_cpu_min = np.min(torch_cpu_times) * 1000
    print(f"  torch CPU       avg={torch_cpu_avg:10.3f} ms   best={torch_cpu_min:10.3f} ms")

    # ----- MPS GPU ----------------------------------------------
    device = torch.device("mps")
    A_gpu = A_cpu.to(device)
    for _ in range(WARMUP):
        _ = A_gpu.sum().item()
    torch.mps.synchronize()

    mps_times = []
    for _ in range(ITERATIONS):
        start = time.perf_counter()
        _ = A_gpu.sum().item()
        torch.mps.synchronize()
        end = time.perf_counter()
        mps_times.append(end - start)
    mps_avg = np.mean(mps_times) * 1000
    mps_min = np.min(mps_times) * 1000
    print(f"  MPS             avg={mps_avg:10.3f} ms   best={mps_min:10.3f} ms")

    print()
    print(f"  naive vs torch CPU:  {naive_avg / torch_cpu_avg:8.1f}x slower")
    print(f"  torch CPU vs MPS:    {torch_cpu_avg / mps_avg:8.2f}x")
    print()