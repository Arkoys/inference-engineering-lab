
import time
import torch


def tiled_matmul(A, B, tile_size):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2
    assert M % tile_size == 0
    assert N % tile_size == 0
    assert K % tile_size == 0

    C = torch.zeros((M, N), dtype=A.dtype, device=A.device)

    for i in range(0, M, tile_size):
        for j in range(0, N, tile_size):
            for k in range(0, K, tile_size):
                A_tile = A[i:i + tile_size, k:k + tile_size]
                B_tile = B[k:k + tile_size, j:j + tile_size]
                C[i:i + tile_size, j:j + tile_size] += A_tile @ B_tile

    return C


def sync():
    if torch.backends.mps.is_available():
        torch.mps.synchronize()


def benchmark(fn, warmup=2, iterations=5):
    for _ in range(warmup):
        fn()
    sync()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    sync()

    return (time.perf_counter() - start) * 1000 / iterations


def main():
    torch.manual_seed(0)
    device = "mps"
    N = 64

    A = torch.randn(N, N, device=device)
    B = torch.randn(N, N, device=device)

    reference = A @ B
    direct_time = benchmark(lambda: A @ B)

    print(f"Matrices : {N} x {N}")
    print(f"torch.matmul : {direct_time:.3f} ms\n")
    print(f"{'Tuile':>8} | {'Temps (ms)':>12} | {'Ratio / direct':>15}")
    print("-" * 42)

    for tile_size in [16, 32, 64]:
        fn = lambda t=tile_size: tiled_matmul(A, B, t)
        result = fn()
        sync()
        torch.testing.assert_close(result, reference, rtol=1e-4, atol=1e-4)

        elapsed = benchmark(fn)
        print(
            f"{tile_size:>8} | {elapsed:>12.3f} | "
            f"{elapsed / direct_time:>12.1f}x"
        )


if __name__ == "__main__":
    main()
