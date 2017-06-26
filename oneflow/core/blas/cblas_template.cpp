#include "oneflow/core/blas/cblas_template.h"

namespace oneflow {

// level 1 vector and vector
// dot product
template<>
void cblas_dot<float>(
    const int N, const float* X, const int incX,
    const float* Y, const int incY) {
  cblas_sdot(N, X, incX, Y, incY);
}

template<>
void cblas_dot<double>(
    const int N, const double* X, const int incX,
    const double* Y, const int incY) {
  cblas_ddot(N, X, incX, Y, incY);
}

// swap x and y
template<>
void cblas_swap<float>(
    const int N, float* X, const int incX, float* Y, const int incY) {
  cblas_sswap(N, X, incX, Y, incY);
}

template<>
void cblas_swap<double>(
    const int N, double* X, const int incX, double* Y, const int incY) {
  cblas_dswap(N, X, incX, Y, incY);
}

// copy x into y
template<>
void cblas_copy<float>(
    const int N, const float* X, const int incX, float* Y, const int incY) {
  cblas_scopy(N, X, incX, Y, incY);
}

template<>
void cblas_copy<double>(
    const int N, const double* X, const int incX, double* Y, const int incY) {
  cblas_dcopy(N, X, incX, Y, incY);
}

// y = a*x + y
template<>
void cblas_axpy<float>(
    const int N, const float alpha, const float* X, const int incX,
    float* Y, const int incY) {
  cblas_saxpy(N, alpha, X, incX, Y, incY);
}

template<>
void cblas_axpy<double>(
    const int N, const double alpha, const double* X, const int incX,
    double* Y, const int incY) {
  cblas_daxpy(N, alpha, X, incX, Y, incY);
}

// x = a*x
template<>
void cblas_scal<float>(
    const int N, const float alpha, float* X, const int incX) {
  cblas_sscal(N, alpha, X, incX);
}

template<>
void cblas_scal<double>(
    const int N, const double alpha, double* X, const int incX) {
  cblas_dscal(N, alpha, X, incX);
}

// level 2 matrix and vector
// matrix vector multiply
template<>
void cblas_gemv<float>(
    const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
    const int M, const int N, const float alpha,
    const float* A, const int lda,
    const float* X, const int incX, const float beta,
    float* Y, const int incY) {
  cblas_sgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

template<>
void cblas_gemv<double>(
    const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
    const int M, const int N, const double alpha,
    const double* A, const int lda,
    const double* X, const int incX, const double beta,
    double* Y, const int incY) {
  cblas_dgemv(order, TransA, M, N, alpha, A, lda, X, incX, beta, Y, incY);
}

// matrix matrix multiply
template<>
void cblas_gemm<float>(
    const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const int lda,
    const float* B, const int ldb, const float beta,
    float* C, const int ldc) {
  cblas_sgemm(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}

template<>
void cblas_gemm<double>(
    const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const int lda,
    const double* B, const int ldb, const double beta,
    double* C, const int ldc) {
  cblas_dgemm(order, TransA, TransB, M, N, K, alpha, A, lda, B, ldb,
              beta, C, ldc);
}


}  // namespace oneflow
