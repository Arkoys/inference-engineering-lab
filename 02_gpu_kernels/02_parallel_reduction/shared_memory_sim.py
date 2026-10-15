
def shared_memory_reduction(values):
    shared = list(values)
    n = len(shared)
    stride = n // 2
    stage = 0

    print("Valeurs chargées :", shared)

    while stride > 0:
        # Chaque paire active combine deux valeurs.
        for tid in range(stride):
            shared[tid] += shared[tid + stride]

        stage += 1
        print(f"Étape {stage}, stride={stride}: {shared[:stride]}")

        stride //= 2

    print("\nRésultat final :", shared[0])


if __name__ == "__main__":
    shared_memory_reduction([3, 7, 2, 9, 4, 1, 8, 5])
