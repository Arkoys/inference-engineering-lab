
import time
import torch


def benchmark(fn, warmup=5, iterations=20):
    for _ in range(warmup):
        fn()

    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    start = time.perf_counter()

    for _ in range(iterations):
        fn()

    if torch.backends.mps.is_available():
        torch.mps.synchronize()

    return (time.perf_counter() - start) * 1000 / iterations


def main():
    N = 10_000_000
    x = torch.randn(N, device="mps", dtype=torch.float32)

    direct = lambda: x.sum()

    def chunked():
        chunks = x.split(100_000)
        partials = torch.stack([chunk.sum() for chunk in chunks])
        return partials.sum()

    # Vérification numérique
    a = direct()
    b = chunked()
    torch.testing.assert_close(a, b, rtol=1e-4, atol=1e-3)

    t_direct = benchmark(direct)
    t_chunked = benchmark(chunked)

    print(f"Device : {x.device}")
    print(f"Nombre d'éléments : {N:,}")
    print(f"Somme directe : {a.item():.4f}")
    print(f"Somme par blocs : {b.item():.4f}")
    print(f"Directe : {t_direct:.3f} ms")
    print(f"Par blocs : {t_chunked:.3f} ms")
    print(f"Ratio blocs/direct : {t_chunked / t_direct:.2f}x")


if __name__ == "__main__":
    main()
