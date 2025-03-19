#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>

// Basic matrix multiplication
void matrix_multiply_basic(const std::vector<float>& A, 
                          const std::vector<float>& B, 
                          std::vector<float>& C, 
                          int n) {
    // Simple triple loop implementation
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            float sum = 0.0f;
            for (int k = 0; k < n; k++) {
                sum += A[i * n + k] * B[k * n + j];
            }
            C[i * n + j] = sum;
        }
    }
}

// Function to initialize a matrix with random values
void initialize_matrix(std::vector<float>& matrix, int n) {
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
// Function to transpose a matrix
void transpose_matrix(const std::vector<float>& B, std::vector<float>& Bt, int n) {
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            Bt[j * n + i] = B[i * n + j];
        }
    }
}

int main() {
    std::cout << "Matrix Multiplication Performance Test" << std::endl;
    std::cout << "=====================================" << std::endl;
    
    // Get matrix size from user
    int n;
    std::cout << "Enter size for the nxn matrices: ";
    std::cin >> n;
    
    // Create matrices
    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> C(n * n);
    std::vector<float> Bt(n * n);
    
    // Initialize matrices with random values
    initialize_matrix(A, n);
    initialize_matrix(B, n);

    // Transpose matrix B
    transpose_matrix(B, Bt, n);
    
    // Warm-up run (Using Transposed Matrix)
    matrix_multiply_basic(A, Bt, C, n);

    double avggflops = 0;
    for (int i = 0; i < 10; i++) {  // Fix: start from i = 0, run exactly 10 times
        // Timed run
        auto start = std::chrono::high_resolution_clock::now();
        matrix_multiply_basic(A, B, C, n);  // Use Bt instead of B
        auto end = std::chrono::high_resolution_clock::now();
        
        // Calculate execution time
        std::chrono::duration<double> elapsed = end - start;
        double seconds = elapsed.count();
        
        // Calculate GFLOPS
        double gflops = calculate_gflops(n, seconds);
        
        // Display results
        std::cout << "\nMatrix size: " << n << "×" << n << std::endl;
        std::cout << "Data type: float (32-bit)" << std::endl;
        std::cout << "Execution time: " << std::fixed << std::setprecision(4) 
                << seconds * 1000 << " ms" << std::endl;
        std::cout << "Performance: " << std::fixed << std::setprecision(4) 
                << gflops << " GFLOPS" << std::endl;
        
        avggflops += gflops;
    }
    
    std::cout << "Average GFLOPS: " << avggflops / 10 << std::endl;
    return 0;
}