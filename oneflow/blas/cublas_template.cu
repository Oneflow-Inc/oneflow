#include "oneflow/blas/cublas_template.h"

namespace oneflow {

template<>
void cublas_gemm<float>(
    const cublasHandle_t& cublas_handle, const cublasOperation_t cuTransA,
    const cublasOperation_t cuTransB, const int M, const int N, const int K,
    const float* alpha, const float* A, const int lda, const float* B,
    const int ldb, const float* beta, float* C, const int ldc) {
  CHECK_EQ(cublasSgemm(
               cublas_handle, cuTransA, cuTransB, M, N, K, alpha, A, lda, B,
               ldb, beta, C, ldc),
           CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_gemm<double>(
    const cublasHandle_t& cublas_handle, const cublasOperation_t cuTransA,
    const cublasOperation_t cuTransB, const int M, const int N, const int K,
    const double* alpha, const double* A, const int lda, const double* B,
    const int ldb, const double* beta, double* C, const int ldc) {
  CHECK_EQ(cublasDgemm(
               cublas_handle, cuTransA, cuTransB, M, N, K, alpha, A, lda, B,
               ldb, beta, C, ldc),
           CUBLAS_STATUS_SUCCESS);
}

}  // namespace oneflow
