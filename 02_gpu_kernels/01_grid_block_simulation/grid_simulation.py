
import math
import torch


def simulate_kernel(N: int, block_size: int):
    """
    Simule l'organisation d'un kernel CUDA.
    Ce code s'exécute sur le CPU : il ne lance aucun kernel CUDA.
    """
    num_blocks = math.ceil(N / block_size)
    total_threads = num_blocks * block_size

    # Simule l'index calculé par chaque thread.
    indices = []

    for block_idx in range(num_blocks):
        for thread_idx in range(block_size):
            i = block_idx * block_size + thread_idx

            # Comme dans un kernel CUDA, certains threads
            # peuvent dépasser la taille du vecteur.
            if i < N:
                indices.append(i)

    return num_blocks, total_threads, indices


def main():
    N = 1000

    print(f"Taille du vecteur : {N}\n")
    print(f"{'Block size':>12} | {'Blocks':>6} | {'Threads lancés':>15} | {'Threads inutilisés':>18}")
    print("-" * 63)

    for block_size in [32, 64, 128, 256]:
        num_blocks, total_threads, indices = simulate_kernel(N, block_size)

        unused_threads = total_threads - N

        # Vérifie que chaque élément est traité exactement une fois.
        assert len(indices) == N
        assert sorted(indices) == list(range(N))

        print(
            f"{block_size:>12} | "
            f"{num_blocks:>6} | "
            f"{total_threads:>15} | "
            f"{unused_threads:>18}"
        )

    # Vérification numérique avec PyTorch sur le Mac.
    a = torch.arange(N, dtype=torch.float32)
    b = torch.ones(N, dtype=torch.float32)
    c = a + b

    assert torch.equal(c, a + 1)
    print("\nVérification PyTorch réussie.")
    print("Les 5 premiers résultats :", c[:5].tolist())
    print("Attention : cette addition PyTorch n'est pas notre kernel CUDA.")


if __name__ == "__main__":
    main()
