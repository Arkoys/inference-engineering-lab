
def count_naive_reads(N):
    # Chaque sortie C[i,j] relit N valeurs de A et N de B.
    return 2 * N**3


def count_tiled_reads(N, tile_size):
    assert N % tile_size == 0

    tiles_per_dimension = N // tile_size
    output_tiles = tiles_per_dimension**2
    k_tiles = tiles_per_dimension

    # Pour chaque tuile de sortie et chaque étape K :
    # charger une tuile A et une tuile B.
    reads_per_step = 2 * tile_size**2

    return output_tiles * k_tiles * reads_per_step


def main():
    N = 32
    naive_reads = count_naive_reads(N)

    print(f"Matrices carrées : {N} x {N}")
    print(f"Lectures scalaires du modèle naïf : {naive_reads:,}\n")
    print(f"{'Tuile':>8} | {'Lectures estimées':>20} | {'Réduction théorique':>21}")
    print("-" * 57)

    for tile_size in [1, 2, 4, 8, 16, 32]:
        tiled_reads = count_tiled_reads(N, tile_size)
        factor = naive_reads / tiled_reads

        print(
            f"{tile_size:>8} | {tiled_reads:>20,} | "
            f"{factor:>18.1f}x"
        )


if __name__ == "__main__":
    main()
