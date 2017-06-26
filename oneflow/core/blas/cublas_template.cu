#include "oneflow/core/blas/cublas_template.h"

namespace oneflow {

template<>
void cublas_gemm<float>(
    const cublasHandle_t& cublas_handle, cublasOperation_t cuTransA,
    cublasOperation_t cuTransB, int M, int N, int K,
    const float alpha, const float* A, int lda, const float* B,
    int ldb, const float beta, float* C, int ldc) {
  CHECK_EQ(cublasSgemm(
               cublas_handle, cuTransA, cuTransB, M, N, K, &alpha, A, lda, B,
               ldb, &beta, C, ldc),
           CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_gemm<double>(
    const cublasHandle_t& cublas_handle, cublasOperation_t cuTransA,
    cublasOperation_t cuTransB, int M, int N, int K,
    const double alpha, const double* A, int lda, const double* B,
    int ldb, const double beta, double* C, int ldc) {
  CHECK_EQ(cublasDgemm(
               cublas_handle, cuTransA, cuTransB, M, N, K, &alpha, A, lda, B,
               ldb, &beta, C, ldc),
           CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_axpy<float>(
    cublasHandle_t handle, int n,
    const float *alpha,
    const float *x, int incx,
    float *y, int incy) {
  CHECK_EQ(cublasSaxpy(handle, n, alpha, x, incx, y, incy),
           CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_axpy<double>(
    cublasHandle_t handle, int n,
    const double *alpha,
    const double *x, int incx,
    double *y, int incy) {
  CHECK_EQ(cublasDaxpy(handle, n, alpha, x, incx, y, incy),
          CUBLAS_STATUS_SUCCESS);
}

}  // namespace oneflow
