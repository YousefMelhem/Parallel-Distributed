#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <thread>
#include <immintrin.h> // For AVX intrinsics
#include <cstring>  // Add this line for memset

// Define alignment for AVX
#define ALIGN_TO 32

// Aligned matrix multiplication using AVX
void matrix_multiply_avx(const float* A, const float* B, float* C, int n) {
    // Zero out the result matrix
    std::memset(C, 0, n * n * sizeof(float));
    
    // For each row of A
    for (int i = 0; i < n; i++) {
        // For each column of B
        for (int j = 0; j < n; j += 8) {
            // For blocks of 8 columns at a time (AVX register width)
            if (j + 8 <= n) {
                // For each element in the row/column
                for (int k = 0; k < n; k++) {
                    // Broadcast A[i,k] to all elements of the AVX register
                    __m256 a = _mm256_set1_ps(A[i * n + k]);
                    
                    // Load 8 elements from B
                    __m256 b = _mm256_loadu_ps(&B[k * n + j]);
                    
                    // Load current values from C
                    __m256 c = _mm256_loadu_ps(&C[i * n + j]);
                    
                    // Multiply and add
                    c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
                    
                    // Store result back to C
                    _mm256_storeu_ps(&C[i * n + j], c);
                }
            }
            else {
                // Handle edge case where n is not divisible by 8
                for (int jj = j; jj < n; jj++) {
                    float sum = 0.0f;
                    for (int k = 0; k < n; k++) {
                        sum += A[i * n + k] * B[k * n + jj];
                    }
                    C[i * n + jj] = sum;
                }
            }
        }
    }
}

// Multi-threaded AVX matrix multiplication
void matrix_multiply_avx_parallel(const float* A, const float* B, float* C, int n, int num_threads) {
    // Zero out the result matrix
    std::memset(C, 0, n * n * sizeof(float));
    
    // Create threads
    std::vector<std::thread> threads;
    
    // Function for each thread to execute
    auto worker = [&](int start_row, int end_row) {
        // Process assigned rows
        for (int i = start_row; i < end_row; i++) {
            // For each column of B
            for (int j = 0; j < n; j += 8) {
                // For blocks of 8 columns at a time (AVX register width)
                if (j + 8 <= n) {
                    // For each element in the row/column
                    for (int k = 0; k < n; k++) {
                        // Broadcast A[i,k] to all elements of the AVX register
                        __m256 a = _mm256_set1_ps(A[i * n + k]);
                        
                        // Load 8 elements from B
                        __m256 b = _mm256_loadu_ps(&B[k * n + j]);
                        
                        // Load current values from C
                        __m256 c = _mm256_loadu_ps(&C[i * n + j]);
                        
                        // Multiply and add
                        c = _mm256_add_ps(c, _mm256_mul_ps(a, b));
                        
                        // Store result back to C
                        _mm256_storeu_ps(&C[i * n + j], c);
                    }
                }
                else {
                    // Handle edge case where n is not divisible by 8
                    for (int jj = j; jj < n; jj++) {
                        float sum = 0.0f;
                        for (int k = 0; k < n; k++) {
                            sum += A[i * n + k] * B[k * n + jj];
                        }
                        C[i * n + jj] = sum;
                    }
                }
            }
        }
    };
    
    // Divide work among threads
    int rows_per_thread = n / num_threads;
    
    for (int t = 0; t < num_threads; t++) {
        int start_row = t * rows_per_thread;
        int end_row = (t == num_threads - 1) ? n : (t + 1) * rows_per_thread;
        
        threads.emplace_back(worker, start_row, end_row);
    }
    
    // Wait for all threads to complete
    for (auto& thread : threads) {
        thread.join();
    }
}

// Function to initialize a matrix with random values
void initialize_matrix(float* matrix, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    for (int i = 0; i < n * n; i++) {
        matrix[i] = dis(gen);
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

int main() {
    std::cout << "Advanced Matrix Multiplication Performance Test" << std::endl;
    std::cout << "=============================================" << std::endl;
    
    // Get matrix size from user
    int n;
    std::cout << "Enter size for the nxn matrices: ";
    std::cin >> n;
    
    // Align n to be divisible by 8 for AVX
    int aligned_n = ((n + 7) / 8) * 8;
    if (aligned_n != n) {
        std::cout << "Adjusting matrix size to " << aligned_n 
                  << " for better AVX alignment" << std::endl;
        n = aligned_n;
    }
    
    // Allocate aligned memory for matrices
    float* A = allocate_aligned_memory(n * n);
    float* B = allocate_aligned_memory(n * n);
    float* C = allocate_aligned_memory(n * n);
    
    // Initialize matrices with random values
    initialize_matrix(A, n);
    initialize_matrix(B, n);
    
    // Get number of CPU cores
    int num_threads = std::thread::hardware_concurrency();
    std::cout << "\nDetected " << num_threads << " CPU cores/threads" << std::endl;
    
    // Single-threaded AVX implementation
    {
        // Warm-up run
        matrix_multiply_avx(A, B, C, n);
        
        // Timed run
        auto start = std::chrono::high_resolution_clock::now();
        matrix_multiply_avx(A, B, C, n);
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate execution time
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();
        
        // Calculate GFLOPS
        double gflops = calculate_gflops(n, seconds);
        
        // Display results
        std::cout << "\nSingle-threaded AVX implementation:" << std::endl;
        std::cout << "Execution time: " << std::fixed << std::setprecision(4) 
                  << seconds * 1000 << " ms" << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(4) 
                  << gflops << " GFLOPS" << std::endl;
    }
    
    // Multi-threaded AVX implementation
    {
        // Warm-up run
        matrix_multiply_avx_parallel(A, B, C, n, num_threads);
        
        // Timed run
        auto start = std::chrono::high_resolution_clock::now();
        matrix_multiply_avx_parallel(A, B, C, n, num_threads);
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate execution time
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();
        
        // Calculate GFLOPS
        double gflops = calculate_gflops(n, seconds);
        
        // Display results
        std::cout << "\nParallel AVX implementation (" << num_threads << " threads):" << std::endl;
        std::cout << "Execution time: " << std::fixed << std::setprecision(4) 
                  << seconds * 1000 << " ms" << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(4) 
                  << gflops << " GFLOPS" << std::endl;
    }
    
    // Free allocated memory
    free_aligned_memory(A);
    free_aligned_memory(B);
    free_aligned_memory(C);
    
    return 0;
}
