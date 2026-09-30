import torch

N = 1024
BLOCK_SIZE = 32

A = torch.arange(N, dtype=torch.float32)

num_blocks = N // BLOCK_SIZE

partial_sums = []

for block_id in range(num_blocks):
    start = block_id * BLOCK_SIZE
    end = start + BLOCK_SIZE

    block = A[start:end]

    partial_sum = block.sum()

    partial_sums.append(partial_sum)

partial_sums = torch.stack(partial_sums)

print("Number of blocks:", num_blocks)
print("Partial sums:", partial_sums)
print("Final sum:", partial_sums.sum().item())
print("Torch sum:", A.sum().item())