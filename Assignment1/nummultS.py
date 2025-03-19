import numpy as np
from time import monotonic
from threadpoolctl import threadpool_limits

N = 2048
itr = 120
l = 1  # Set thread limit
A = np.random.randn(N, N).astype(np.float32)
B = np.random.randn(N, N).astype(np.float32)
C = np.zeros((N, N), dtype=np.float32)
avg = 0

with threadpool_limits(limits=3):
    for i in range(itr):
        start = monotonic()
        C = A @ B
        end = monotonic()
        s = end - start
        gflops = (N * N * 2 * N) / (s * 10**9)
        avg += gflops
        print(f"GFLOPS: {gflops:.6f}")

    print(f"avg: {avg / itr:.2f}, Thread limit: {l}")