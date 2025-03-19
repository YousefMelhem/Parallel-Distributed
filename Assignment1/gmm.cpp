#include <omp.h>
const int N = 2048; // Matrix size

const int BLOCKSIZE = 4;

float A[N][N], B[N][N], C[N][N], Cvals[N][N];
void gemm_omp_ikj() {
    int bi, bk, bj, i, k, j;

    #pragma omp parallel for private(bk, bj, i, k, j) shared(A, B, C)
    for(bi = 0; bi < N; bi += BLOCKSIZE)
        for(bk = 0; bk < N; bk += BLOCKSIZE)
            for(bj = 0; bj < N; bj += BLOCKSIZE)

                for(i = 0; i < BLOCKSIZE; i++)
                    for(k = 0; k < BLOCKSIZE; k++) {
                        float a_val = A[bi + i][bk + k];
                        for(j = 0; j < BLOCKSIZE; j+=2) {
                            C[bi + i][bj + j + 0] += a_val * B[bk + k][bj + j + 0];
                            C[bi + i][bj + j + 1] += a_val * B[bk + k][bj + j + 1];
                        }
                    }
}


