
def tree_reduction(values):
    active = list(values)
    stage = 0
    total_additions = 0

    print(f"Départ : {active}")

    while len(active) > 1:
        next_active = []

        for i in range(0, len(active), 2):
            next_active.append(active[i] + active[i + 1])
            total_additions += 1

        active = next_active
        stage += 1

        print(f"Étape {stage} : {active}")

    print(f"\nRésultat : {active[0]}")
    print(f"Étapes : {stage}")
    print(f"Additions : {total_additions}")


if __name__ == "__main__":
    tree_reduction(list(range(1, 33)))
