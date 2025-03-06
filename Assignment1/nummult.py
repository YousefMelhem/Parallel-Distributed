import numpy as np
import time

# Include powers of 2 and some nearby sizes
sizes = [512, 1000, 1024, 1536, 2000, 2048, 3000, 4000, 4096]
results = []

for n in sizes:
    # Create matrices
    A = np.random.rand(n, n).astype(np.float32)
    B = np.random.rand(n, n).astype(np.float32)
    
    # Warm-up run
    _ = np.matmul(A, B)
    
    # Timed runs
    times = []
    for _ in range(3):  # Average of 3 runs
        start = time.time()
        C = np.matmul(A, B)
        end = time.time()
        times.append(end - start)
    
    avg_time = sum(times) / len(times)
    flops = 2 * (n**3)
    gflops = flops / avg_time / 1e9
    
    results.append((n, avg_time, gflops))
    print(f"Size: {n}×{n}, Time: {avg_time*1000:.2f} ms, Performance: {gflops:.2f} GFLOPS")

# Find optimal size
optimal = max(results, key=lambda x: x[2])
print(f"\nOptimal matrix size: {optimal[0]}×{optimal[0]} with {optimal[2]:.2f} GFLOPS")