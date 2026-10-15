
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
            # Tuile de sortie C[i:i+T, j:j+T]
            for k in range(0, K, tile_size):
                A_tile = A[i:i+tile_size, k:k+tile_size]
                B_tile = B[k:k+tile_size, j:j+tile_size]

                # Multiplication de deux petites tuiles.
                # torch.matmul réalise ici le calcul local.
                C[i:i+tile_size, j:j+tile_size] += A_tile @ B_tile

    return C


def main():
    torch.manual_seed(0)
    device = "mps"
    M = K = N = 32

    A = torch.randn(M, K, device=device)
    B = torch.randn(K, N, device=device)

    reference = A @ B

    for tile_size in [1, 2, 4, 8, 16, 32]:
        result = tiled_matmul(A, B, tile_size)
        torch.testing.assert_close(
            result, reference, rtol=1e-4, atol=1e-4
        )
        print(f"Tuile {tile_size:2}x{tile_size:2} : résultat correct")

    print("\nToutes les configurations sont correctes.")


if __name__ == "__main__":
    main()
