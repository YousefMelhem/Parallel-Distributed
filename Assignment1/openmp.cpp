#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <omp.h>  // Include OpenMP

// Optimized matrix multiplication with blocking & OpenMP
void matrix_multiply_optimized(const std::vector<float>& A, 
                              const std::vector<float>& B, 
                              std::vector<float>& C, 
                              int n, int block_size) {
    
    // Zero out the result matrix
    std::fill(C.begin(), C.end(), 0.0f);
    
    // Blocked matrix multiplication with OpenMP
    #pragma omp parallel for collapse(2)  // Parallelize loops
    for (int i = 0; i < n; i += block_size) {
        for (int j = 0; j < n; j += block_size) {
            for (int k = 0; k < n; k += block_size) {
                for (int ii = i; ii < std::min(i + block_size, n); ii++) {
                    for (int jj = j; jj < std::min(j + block_size, n); jj++) {
                        float sum = C[ii * n + jj];
                        for (int kk = k; kk < std::min(k + block_size, n); kk++) {
                            sum += A[ii * n + kk] * B[jj * n + kk]; // Use transposed B
                        }
                        C[ii * n + jj] = sum;
                    }
                }
            }
        }
    }
}

// Function to transpose a matrix
void transpose_matrix(const std::vector<float>& src, std::vector<float>& dst, int n) {
    #pragma omp parallel for collapse(2)  // Parallelize transpose
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            dst[j * n + i] = src[i * n + j];
        }
    }
}

// Function to initialize a matrix with random values
void initialize_matrix(std::vector<float>& matrix, int n) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<float> dis(0.0f, 1.0f);
    
    #pragma omp parallel for  // Parallelize initialization
    for (int i = 0; i < n * n; i++) {
        matrix[i] = dis(gen);
    }
}

// Function to calculate GFLOPS
double calculate_gflops(int n, double seconds) {
    double operations = 2.0 * std::pow(n, 3);
    return operations / seconds / 1e9;
}

int main() {
    int opti_block_size;
    double optimal_gflops = 0;
    
    std::cout << "Matrix Multiplication Performance Test (OpenMP Enabled)\n";
    std::cout << "=====================================================\n";
    
    // Get matrix size from user
    int n;
    std::cout << "Enter size for the nxn matrices: ";
    std::cin >> n;
    
    // Define block size
    const int max_block_size = 64;

    // Create matrices
    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> B_T(n * n); // Transposed matrix
    std::vector<float> C(n * n);
    
    // Initialize matrices
    initialize_matrix(A, n);
    initialize_matrix(B, n);
    
    // Transpose matrix B
    transpose_matrix(B, B_T, n);
    
    for(int block_size = 2; block_size <= max_block_size; block_size *= 2) {
        // Warm-up run
        matrix_multiply_optimized(A, B_T, C, n, block_size);
    
        // Timed run
        auto start = std::chrono::high_resolution_clock::now();
        matrix_multiply_optimized(A, B_T, C, n, block_size);  // Use transposed matrix
        auto end = std::chrono::high_resolution_clock::now();
    
        // Calculate execution time
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();
    
        // Calculate GFLOPS
        double gflops = calculate_gflops(n, seconds);

        if (gflops > optimal_gflops) {
            opti_block_size = block_size;
            optimal_gflops = gflops;
        }
    }

    std::cout << "Optimal block size: " << opti_block_size 
              << ", Best performance: " << optimal_gflops << " GFLOPS\n";
    
    return 0;
}
