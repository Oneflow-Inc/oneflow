#ifndef ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_
#define ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

// level 1 vector and vector
// dot product
template<typename FloatingPointType>
void cublas_dot(
    cublasHandle_t handle, int n,
    const FloatingPointType* x, int incx,
    const FloatingPointType* y, int incy, FloatingPointType* result);

// swap x and y
template<typename FloatingPointType>
void cublas_swap(
    cublasHandle_t handle, int n,
    FloatingPointType* x, int incx, FloatingPointType* y, int incy);

// copy x into y
template<typename FloatingPointType>
void cublas_copy(
    cublasHandle_t handle, int n,
    const FloatingPointType* x, int incx,
    FloatingPointType* y, int incy);

// y = a*x + y
template<typename FloatingPointType>
void cublas_axpy(
    cublasHandle_t handle, int n,
    const FloatingPointType* alpha,
    const FloatingPointType* x, int incx,
    FloatingPointType* y, int incy);

// x = a*x
template<typename FloatingPointType>
void cublas_scal(
    cublasHandle_t handle, int n,
    const FloatingPointType* alpha, FloatingPointType* x, int incx);

// level 2 matrix and vector
// matrix vector multiply
template<typename FloatingPointType>
void cublas_gemv(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const FloatingPointType* alpha, const FloatingPointType* a, int lda,
    const FloatingPointType* x, int incx, const FloatingPointType* beta,
    FloatingPointType* y, int incy);

// level 3 matrix and matrix
// matrix matrix multiply
template<typename FloatingPointType>
void cublas_gemm(
    cublasHandle_t handle, cublasOperation_t cutrans_a,
    cublasOperation_t cutrans_b, int m, int n, int k,
    const FloatingPointType* alpha, const FloatingPointType* a, int lda,
    const FloatingPointType* b, int ldb,
    const FloatingPointType* beta, FloatingPointType* c, int ldc);

} // namespace oneflow

#endif // ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_
