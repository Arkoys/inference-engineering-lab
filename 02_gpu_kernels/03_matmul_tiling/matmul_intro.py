
import time
import torch


def naive_matmul(A, B):
    M, K = A.shape
    K2, N = B.shape
    assert K == K2

    C = torch.zeros((M, N), dtype=A.dtype, device=A.device)

    for i in range(M):
        for j in range(N):
            for k in range(K):
                C[i, j] += A[i, k] * B[k, j]

    return C


def benchmark(fn, warmup=3, iterations=10):
    for _ in range(warmup):
        fn()

    torch.mps.synchronize()

    start = time.perf_counter()
    for _ in range(iterations):
        fn()
    torch.mps.synchronize()

    return (time.perf_counter() - start) * 1000 / iterations


def main():
    device = "mps"
    M, K, N = 32, 32, 32

    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)

    C_naive = naive_matmul(A, B)
    C_torch = torch.matmul(A, B)

    torch.testing.assert_close(C_naive, C_torch, rtol=1e-4, atol=1e-4)

    print(f"Tailles : A={tuple(A.shape)}, B={tuple(B.shape)}")
    print("Résultats corrects :", torch.allclose(C_naive, C_torch, rtol=1e-4, atol=1e-4))
    print(f"Naïve : {benchmark(lambda: naive_matmul(A, B)):.3f} ms")
    print(f"torch.matmul : {benchmark(lambda: torch.matmul(A, B)):.3f} ms")


if __name__ == "__main__":
    main()
