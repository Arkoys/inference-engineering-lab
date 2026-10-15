
import math


def simulate_grid_stride(N: int, block_size: int, num_blocks: int):
    total_threads = block_size * num_blocks
    elements_per_thread = [0] * total_threads
    processed_indices = []

    # Simulation CPU du motif grid-stride CUDA.
    for thread_id in range(total_threads):
        i = thread_id

        while i < N:
            processed_indices.append(i)
            elements_per_thread[thread_id] += 1
            i += total_threads

    assert sorted(processed_indices) == list(range(N))
    assert len(processed_indices) == N

    active_threads = sum(count > 0 for count in elements_per_thread)
    max_elements = max(elements_per_thread)

    return active_threads, max_elements, elements_per_thread


def main():
    N = 1000
    block_size = 128

    print(f"N = {N}, block_size = {block_size}\n")
    print(
        f"{'Blocks':>6} | {'Threads':>7} | {'Threads actifs':>14} | "
        f"{'Max éléments/thread':>19}"
    )
    print("-" * 58)

    for num_blocks in [1, 2, 4, 8, 16]:
        active, max_elements, _ = simulate_grid_stride(
            N, block_size, num_blocks
        )
        total_threads = block_size * num_blocks

        print(
            f"{num_blocks:>6} | {total_threads:>7} | "
            f"{active:>14} | {max_elements:>19}"
        )

    print("\nToutes les configurations traitent exactement les 1000 éléments.")


if __name__ == "__main__":
    main()
