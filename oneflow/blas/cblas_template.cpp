#include "oneflow/blas/cblas_template.h"

namespace oneflow {

template<>
void cblas_gemm<float>(
    const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const float alpha, const float* A,
    const int lda, const float* B, const int ldb, const float beta, float* C,
    const int ldc) {
  cblas_sgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template<>
void cblas_gemm<double>(
    const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const double alpha, const double* A,
    const int lda, const double* B, const int ldb, const double beta, double* C,
    const int ldc) {
  cblas_dgemm(CblasRowMajor, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template<>
void cblas_axpy<float>(
    const int N,
    const float alpha,
    const float *x, const int incx,
    float *y, const int incy) {
  cblas_saxpy(N, alpha, x, incx, y, incy);
}

template<>
void cblas_axpy<double>(
    const int N,
    const double alpha,
    const double *x, const int incx,
    double *y, const int incy) {
  cblas_daxpy(N, alpha, x, incx, y, incy);
}

}  // namespace oneflow
