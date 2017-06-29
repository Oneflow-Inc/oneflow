#include "oneflow/core/blas/cublas_template.h"

namespace oneflow {

// level 1 vector and vector
// dot product
template<>
void cublas_dot<float>(
    cublasHandle_t handle, int n, const float* x, int incx, const float* y,
    int incy, float* result) {
  CHECK_EQ(cublasSdot(handle, n, x, incx, y, incy, result),
           CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_dot<double>(
    cublasHandle_t handle, int n, const double* x, int incx, const double* y,
    int incy, double* result) {
  CHECK_EQ(cublasDdot(handle, n, x, incx, y, incy, result),
           CUBLAS_STATUS_SUCCESS);
}

// swap x and y
template<>
void cublas_swap<float>(
    cublasHandle_t handle, int n, float* x, int incx, float* y, int incy) {
  CHECK_EQ(cublasSswap(handle, n, x, incx, y, incy), CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_swap<double>(
    cublasHandle_t handle, int n, double* x, int incx, double* y, int incy) {
  CHECK_EQ(cublasDswap(handle, n, x, incx, y, incy), CUBLAS_STATUS_SUCCESS);
}

// copy x into y
template<>
void cublas_copy<float>(
    cublasHandle_t handle, int n, const float* x, int incx,
    float* y, int incy) {
  CHECK_EQ(cublasScopy(handle, n, x, incx, y, incy), CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_copy<double>(
    cublasHandle_t handle, int n, const double* x, int incx,
    double* y, int incy) {
  CHECK_EQ(cublasDcopy(handle, n, x, incx, y, incy), CUBLAS_STATUS_SUCCESS);
}

// y = a*x + y
template<>
void cublas_axpy<float>(
    cublasHandle_t handle, int n, const float* alpha, const float* x, 
    const int incx, float* y, const int incy) {
  CHECK_EQ(cublasSaxpy(handle, n, alpha, x, incx, y, incy),
           CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_axpy<double>(
    cublasHandle_t handle, int n, const double* alpha, 
    const double* x, const int incx,
    double* y, int incy) {
  CHECK_EQ(cublasDaxpy(handle, n, alpha, x, incx, y, incy),
           CUBLAS_STATUS_SUCCESS);
}

// x = a*x
template<>
void cublas_scal<float>(
    cublasHandle_t handle, int n, const float* alpha, float* x, int incx) {
  CHECK_EQ(cublasSscal(handle, n, alpha, x, incx), CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_scal<double>(
    cublasHandle_t handle, int n, const double* alpha, double* x, int incx) {
  CHECK_EQ(cublasDscal(handle, n, alpha, x, incx), CUBLAS_STATUS_SUCCESS);
}

// level 2 matrix and vector
// matrix vector multiply
template<>
void cublas_gemv<float>(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const float* alpha, const float* a, int lda, const float* x, int incx,
    const float* beta, float* y, int incy) {
  CHECK_EQ(cublasSgemv(
               handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy),
           CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_gemv<double>(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const double* alpha, const double* a, int lda, const double* x, int incx,
    const double* beta, double* y, int incy) {
  CHECK_EQ(cublasDgemv(
               handle, trans, m, n, alpha, a, lda, x, incx, beta, y, incy),
           CUBLAS_STATUS_SUCCESS);
}

// level 3 matrix and matrix
// matrix matrix multiply
template<>
void cublas_gemm<float>(
    cublasHandle_t handle, cublasOperation_t cutrans_a,
    cublasOperation_t cutrans_b, int m, int n, int k,
    const float* alpha, const float* a, int lda,
    const float* b, int ldb, const float* beta, float* c, int ldc) {
  CHECK_EQ(cublasSgemm(
               handle, cutrans_a, cutrans_b, m, n, k, alpha, a, lda, b, ldb, beta,
               c, ldc),
           CUBLAS_STATUS_SUCCESS);
}

template<>
void cublas_gemm<double>(
    cublasHandle_t handle, cublasOperation_t cutrans_a,
    cublasOperation_t cutrans_b, int m, int n, int k,
    const double* alpha, const double* a, int lda,
    const double* b, int ldb, const double* beta, double* c, int ldc) {
  CHECK_EQ(cublasDgemm(
               handle, cutrans_a, cutrans_b, m, n, k, alpha, a, lda, b, ldb, beta,
               c, ldc),
           CUBLAS_STATUS_SUCCESS);
}

}  // namespace oneflow
