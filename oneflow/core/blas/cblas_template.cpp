#include "oneflow/core/blas/cblas_template.h"

namespace oneflow {

// level 1 vector and vector
// dot product
template<>
float cblas_dot<float>(const int n, const float* x, const int incx,
                       const float* y, const int incy) {
  return cblas_sdot(n, x, incx, y, incy);
}

template<>
double cblas_dot<double>(const int n, const double* x, const int incx,
                         const double* y, const int incy) {
  return cblas_ddot(n, x, incx, y, incy);
}

// swap x and y
template<>
void cblas_swap<float>(const int n, float* x, const int incx, float* y,
                       const int incy) {
  cblas_sswap(n, x, incx, y, incy);
}

template<>
void cblas_swap<double>(const int n, double* x, const int incx, double* y,
                        const int incy) {
  cblas_dswap(n, x, incx, y, incy);
}

// copy x into y
template<>
void cblas_copy<float>(const int n, const float* x, const int incx, float* y,
                       const int incy) {
  cblas_scopy(n, x, incx, y, incy);
}

template<>
void cblas_copy<double>(const int n, const double* x, const int incx, double* y,
                        const int incy) {
  cblas_dcopy(n, x, incx, y, incy);
}

// y = a*x + y
template<>
void cblas_axpy<float>(const int n, const float alpha, const float* x,
                       const int incx, float* y, const int incy) {
  cblas_saxpy(n, alpha, x, incx, y, incy);
}

template<>
void cblas_axpy<double>(const int n, const double alpha, const double* x,
                        const int incx, double* y, const int incy) {
  cblas_daxpy(n, alpha, x, incx, y, incy);
}

// x = a*x
template<>
void cblas_scal<float>(const int n, const float alpha, float* x,
                       const int incx) {
  cblas_sscal(n, alpha, x, incx);
}

template<>
void cblas_scal<double>(const int n, const double alpha, double* x,
                        const int incx) {
  cblas_dscal(n, alpha, x, incx);
}

// level 2 matrix and vector
// matrix vector multiply
template<>
void cblas_gemv<float>(const enum CBLAS_ORDER order,
                       const enum CBLAS_TRANSPOSE trans_a, const int m,
                       const int n, const float alpha, const float* a,
                       const int lda, const float* x, const int incx,
                       const float beta, float* y, const int incy) {
  cblas_sgemv(order, trans_a, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

template<>
void cblas_gemv<double>(const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE trans_a, const int m,
                        const int n, const double alpha, const double* a,
                        const int lda, const double* x, const int incx,
                        const double beta, double* y, const int incy) {
  cblas_dgemv(order, trans_a, m, n, alpha, a, lda, x, incx, beta, y, incy);
}

// matrix matrix multiply
template<>
void cblas_gemm<float>(const enum CBLAS_ORDER order,
                       const enum CBLAS_TRANSPOSE trans_a,
                       const enum CBLAS_TRANSPOSE trans_b, const int m,
                       const int n, const int k, const float alpha,
                       const float* a, const int lda, const float* b,
                       const int ldb, const float beta, float* c,
                       const int ldc) {
  cblas_sgemm(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c,
              ldc);
}

template<>
void cblas_gemm<double>(const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE trans_a,
                        const enum CBLAS_TRANSPOSE trans_b, const int m,
                        const int n, const int k, const double alpha,
                        const double* a, const int lda, const double* b,
                        const int ldb, const double beta, double* c,
                        const int ldc) {
  cblas_dgemm(order, trans_a, trans_b, m, n, k, alpha, a, lda, b, ldb, beta, c,
              ldc);
}

}  // namespace oneflow
