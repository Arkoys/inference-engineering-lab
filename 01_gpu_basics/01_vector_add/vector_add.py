import time
import numpy as np
import torch

sizes = [
    1_000,
    10_000,
    100_000,
    1_000_000,
    10_000_000,
    100_000_000,
]

for N in sizes:
    # benchmark CPU
    # benchmark MPS
    # print results



ITERATIONS = 100

print(f"N = {N:,}")
print(f"Iterations = {ITERATIONS}")
print()

# ============================================================
# CPU
# ============================================================

A_cpu = np.random.rand(N).astype(np.float32)
B_cpu = np.random.rand(N).astype(np.float32)

# Warm-up
for _ in range(10):
    C_cpu = A_cpu + B_cpu

cpu_times = []

for _ in range(ITERATIONS):
    start = time.perf_counter()

    C_cpu = A_cpu + B_cpu

    end = time.perf_counter()

    cpu_times.append(end - start)

cpu_avg = np.mean(cpu_times)
cpu_min = np.min(cpu_times)

print("CPU")
print(f"  Average: {cpu_avg * 1000:.3f} ms")
print(f"  Best:    {cpu_min * 1000:.3f} ms")


# ============================================================
# MPS GPU
# ============================================================

device = torch.device("mps")

A_gpu = torch.rand(N, device=device)
B_gpu = torch.rand(N, device=device)

# Warm-up
for _ in range(10):
    C_gpu = A_gpu + B_gpu

torch.mps.synchronize()

mps_times = []

for _ in range(ITERATIONS):
    start = time.perf_counter()

    C_gpu = A_gpu + B_gpu

    torch.mps.synchronize()

    end = time.perf_counter()

    mps_times.append(end - start)

mps_avg = np.mean(mps_times)
mps_min = np.min(mps_times)

print()
print("MPS")
print(f"  Average: {mps_avg * 1000:.3f} ms")
print(f"  Best:    {mps_min * 1000:.3f} ms")


# ============================================================
# Comparison
# ============================================================

speedup = cpu_avg / mps_avg

print()
print("Comparison")
print(f"  Average speedup: {speedup:.2f}x")

