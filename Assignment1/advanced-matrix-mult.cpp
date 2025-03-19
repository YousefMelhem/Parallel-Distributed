#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <immintrin.h> // For AVX intrinsics
#include <cstring>     // For memset
#include <omp.h>       // For OpenMP

// Define alignment for AVX
#define ALIGN_TO 32

// Cache blocking parameters - adjusted for your specific CPU
#define BLOCK_SIZE 48  // Optimized for your L1 cache size of 1.1MB

// Unroll factor for the innermost loop
#define UNROLL_FACTOR 4

// Matrix multiplication using OpenMP, AVX, cache blocking and loop unrolling
void matrix_multiply_optimized(const float* A, const float* B, float* C, int n, int num_threads) {
    // Set the number of threads for OpenMP
    omp_set_num_threads(num_threads);
    
    // Zero out the result matrix
    std::memset(C, 0, n * n * sizeof(float));
    
    // Cache blocking approach with loop unrolling
    #pragma omp parallel
    {
        // Each thread will have its own private copy of these variables
        int i, j, k, ii, jj, kk;
        
        // Parallelize the blocked computation
        #pragma omp for schedule(dynamic)
        for (ii = 0; ii < n; ii += BLOCK_SIZE) {
            for (kk = 0; kk < n; kk += BLOCK_SIZE) {
                for (jj = 0; jj < n; jj += BLOCK_SIZE) {
                    // Process blocks to maximize cache efficiency
                    for (i = ii; i < std::min(ii + BLOCK_SIZE, n); i++) {
                        for (k = kk; k < std::min(kk + BLOCK_SIZE, n); k++) {
                            // Broadcast A[i,k] to all elements of the AVX register
                            __m256 a = _mm256_set1_ps(A[i * n + k]);
                            
                            // Process mini-blocks with AVX and loop unrolling
                            // Make sure j doesn't exceed the matrix bounds
                            int j_limit = std::min(jj + BLOCK_SIZE, n);
                            
                            // Main loop with unrolling - process UNROLL_FACTOR * 8 elements at once
                            for (j = jj; j + (UNROLL_FACTOR * 8) <= j_limit; j += (UNROLL_FACTOR * 8)) {
                                // Prefetch for better cache behavior
                                #ifdef __GNUC__
                                __builtin_prefetch(&B[k * n + j + 64], 0, 3);
                                __builtin_prefetch(&C[i * n + j + 64], 1, 3);
                                #endif
                                
                                // Unrolled AVX operations - this is the performance hotspot
                                // Unroll 1
                                __m256 b0 = _mm256_loadu_ps(&B[k * n + j]);
                                __m256 c0 = _mm256_loadu_ps(&C[i * n + j]);
                                c0 = _mm256_add_ps(c0, _mm256_mul_ps(a, b0));
                                _mm256_storeu_ps(&C[i * n + j], c0);
                                
                                // Unroll 2
                                __m256 b1 = _mm256_loadu_ps(&B[k * n + j + 8]);
                                __m256 c1 = _mm256_loadu_ps(&C[i * n + j + 8]);
                                c1 = _mm256_add_ps(c1, _mm256_mul_ps(a, b1));
                                _mm256_storeu_ps(&C[i * n + j + 8], c1);
                                
                                // Unroll 3
                                __m256 b2 = _mm256_loadu_ps(&B[k * n + j + 16]);
                                __m256 c2 = _mm256_loadu_ps(&C[i * n + j + 16]);
                                c2 = _mm256_add_ps(c2, _mm256_mul_ps(a, b2));
                                _mm256_storeu_ps(&C[i * n + j + 16], c2);
                                
                                // Unroll 4
                                __m256 b3 = _mm256_loadu_ps(&B[k * n + j + 24]);
                                __m256 c3 = _mm256_loadu_ps(&C[i * n + j + 24]);
                                c3 = _mm256_add_ps(c3, _mm256_mul_ps(a, b3));
                                _mm256_storeu_ps(&C[i * n + j + 24], c3);
                            }
                            
                            // Handle remaining elements with normal AVX (no unrolling)
                            for (; j + 8 <= j_limit; j += 8) {
                                __m256 b = _mm256_loadu_ps(&B[k * n + j]);
                                __m256 c = _mm256_loadu_ps(&C[i * n + j]);
                                c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
                                _mm256_storeu_ps(&C[i * n + j], c);
                            }
                            
                            // Handle remaining elements (less than 8) with scalar operations
                            for (; j < j_limit; j++) {
                                C[i * n + j] += A[i * n + k] * B[k * n + j];
                            }
                        }
                    }
                }
            }
        }
    }
}

// Function to initialize a matrix with random values
void initialize_matrix(float* matrix, int n) {
    // Using a different random approach for better parallel behavior
    #pragma omp parallel
    {
        // Each thread gets its own random generator to avoid contention
        unsigned int seed = omp_get_thread_num() * 1000 + time(NULL);
        
        #pragma omp for
        for (int i = 0; i < n * n; i++) {
            matrix[i] = static_cast<float>(rand_r(&seed)) / RAND_MAX;
        }
    }
}

// Function to calculate GFLOPS
double calculate_gflops(int n, double seconds) {
    // For n×n matrix multiplication: 2*n³ floating point operations
    double operations = 2.0 * std::pow(n, 3);
    return operations / seconds / 1e9;
}

// Allocate aligned memory
float* allocate_aligned_memory(int size) {
    void* mem = _mm_malloc(size * sizeof(float), ALIGN_TO);
    return reinterpret_cast<float*>(mem);
}

// Free aligned memory
void free_aligned_memory(float* ptr) {
    _mm_free(ptr);
}

// Simple verification function to check if the result is correct
bool verify_result(const float* A, const float* B, const float* C, int n) {
    // Create a small test matrix for verification
    const int test_size = std::min(4, n);
    float* test_result = new float[test_size * test_size]();
    
    // Compute a small portion using the standard method
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < test_size; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            test_result[i * test_size + j] = sum;
        }
    }
    
    // Compare with our implementation
    bool is_correct = true;
    const float epsilon = 1e-3f; // Allow for small floating-point differences
    for (int i = 0; i < test_size; i++) {
        for (int j = 0; j < test_size; j++) {
            float diff = std::abs(test_result[i * test_size + j] - C[i * n + j]);
            if (diff > epsilon) {
                std::cout << "Verification failed at position [" << i << "," << j << "]: "
                          << "Expected " << test_result[i * test_size + j] 
                          << ", got " << C[i * n + j] << std::endl;
                is_correct = false;
                break;
            }
        }
        if (!is_correct) break;
    }
    
    delete[] test_result;
    return is_correct;
}

int main() {
    std::cout << "High-Performance Matrix Multiplication Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Get matrix size from user
    int n;
    std::cout << "Enter size for the nxn matrices: ";
    std::cin >> n;
    
    // Align n to be divisible by 32 for better AVX alignment with unrolling
    int aligned_n = ((n + 31) / 32) * 32;
    if (aligned_n != n) {
        std::cout << "Adjusting matrix size to " << aligned_n 
                  << " for better alignment with unrolled AVX code" << std::endl;
        n = aligned_n;
    }
    
    // Allocate aligned memory for matrices
    float* A = allocate_aligned_memory(n * n);
    float* B = allocate_aligned_memory(n * n);
    float* C = allocate_aligned_memory(n * n);
    
    // Initialize matrices with random values
    initialize_matrix(A, n);
    initialize_matrix(B, n);
    
    // Display system information
    int num_threads = omp_get_max_threads();
    std::cout << "\nSystem information:" << std::endl;
    std::cout << "- CPU threads: " << num_threads << std::endl;
    std::cout << "- Cache block size: " << BLOCK_SIZE << std::endl;
    std::cout << "- Unroll factor: " << UNROLL_FACTOR << std::endl;
    std::cout << "- Matrix size: " << n << "x" << n << " (" << (n*n*sizeof(float))/1048576.0 << " MB per matrix)" << std::endl;
    
    // Run optimized version with measurement
    {
        // Warm-up run (important for accurate benchmarking)
        matrix_multiply_optimized(A, B, C, n, num_threads);
        
        // Timed run
        auto start = std::chrono::high_resolution_clock::now();
        
        // Multiple runs for more stable measurements
        const int NUM_RUNS = 3;
        for (int run = 0; run < NUM_RUNS; run++) {
            matrix_multiply_optimized(A, B, C, n, num_threads);
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate average execution time
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count() / NUM_RUNS;
        
        // Calculate GFLOPS
        double gflops = calculate_gflops(n, seconds);
        
        // Display results
        std::cout << "\nOptimized implementation (" << num_threads << " threads):" << std::endl;
        std::cout << "Execution time: " << std::fixed << std::setprecision(2) 
                  << seconds * 1000 << " ms" << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(2) 
                  << gflops << " GFLOPS" << std::endl;
        std::cout << "Efficiency: " << std::fixed << std::setprecision(2)
                  << (gflops / num_threads) << " GFLOPS per thread" << std::endl;
        
        // Verify the result
        bool is_correct = verify_result(A, B, C, n);
        std::cout << "Result verification: " << (is_correct ? "PASSED" : "FAILED") << std::endl;
    }
    
    // Free allocated memory
    free_aligned_memory(A);
    free_aligned_memory(B);
    free_aligned_memory(C);
    
    return 0;
}