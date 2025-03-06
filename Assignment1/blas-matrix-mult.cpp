#include <iostream>
#include <vector>
#include <chrono>
#include <random>
#include <iomanip>
#include <cblas.h> // OpenBLAS header
#include <thread>

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

int main() {
    std::cout << "Matrix Multiplication using OpenBLAS" << std::endl;
    std::cout << "==================================" << std::endl;
    
    // Get matrix size from user
    int n;
    std::cout << "Enter size for the nxn matrices: ";
    std::cin >> n;
    
    // Create matrices
    std::vector<float> A(n * n);
    std::vector<float> B(n * n);
    std::vector<float> C(n * n);
    
    // Initialize matrices with random values
    initialize_matrix(A, n);
    initialize_matrix(B, n);
    
    // Set number of threads for OpenBLAS
    // Get number of CPU cores
    int num_threads = std::thread::hardware_concurrency();
    openblas_set_num_threads(num_threads);
    
    std::cout << "\nUsing OpenBLAS with " << num_threads << " threads" << std::endl;
    
    // Warm-up run
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
    
    // Timed run
    auto start = std::chrono::high_resolution_clock::now();
    
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans,
                n, n, n, 1.0f, A.data(), n, B.data(), n, 0.0f, C.data(), n);
    
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
    
    return 0;
}
