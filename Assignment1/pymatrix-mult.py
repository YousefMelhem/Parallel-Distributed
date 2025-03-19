import time
import numpy as np
n = 256

# Initialize matrices A, B, and C
A = np.random.rand(n, n).astype(np.float32)
B = np.random.rand(n, n).astype(np.float32)
C = np.zeros((n, n), dtype=np.float32)

def matrix_mult(A, B, C):
    for i in range(n):
        for j in range(n):
            for k in range(n):
                C[i][j] += A[i][k] * B[k][j]


#clacluate the time
start = time.time()
matrix_mult(A, B, C)
end = time.time()

# Calculate flop
flops = 2 * n**3
Gflops = flops / (end - start) / 10**9
print(f"Gflops: {Gflops:.6f}")
print(f"Time: {end - start:.6f}")
print(f"Matrix Size: {n}x{n} seconds")