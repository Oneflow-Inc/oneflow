#ifndef ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_
#define ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

// level 1 vector and vector
// dot product
template<typename floating_point_type>
void cublas_dot(
    cublasHandle_t handle, int n,
    const floating_point_type* x, int incx,
    const floating_point_type* y, int incy, floating_point_type* result);

// swap x and y
template<typename floating_point_type>
void cublas_swap(
    cublasHandle_t handle, int n,
    floating_point_type* x, int incx, floating_point_type* y, int incy);

// copy x into y
template<typename floating_point_type>
void cublas_copy(
    cublasHandle_t handle, int n,
    const floating_point_type* x, int incx,
    floating_point_type* y, int incy);

// y = a*x + y
template<typename floating_point_type>
void cublas_axpy(
    cublasHandle_t handle, int n,
    const floating_point_type* alpha,
    const floating_point_type* x, int incx,
    floating_point_type* y, int incy);

// x = a*x
template<typename floating_point_type>
void cublas_scal(
    cublasHandle_t handle, int n,
    const floating_point_type* alpha, floating_point_type* x, int incx);

// level 2 matrix and vector
// matrix vector multiply
template<typename floating_point_type>
void cublas_gemv(
    cublasHandle_t handle, cublasOperation_t trans, int m, int n,
    const floating_point_type* alpha, const floating_point_type* a, int lda,
    const floating_point_type* x, int incx, const floating_point_type* beta,
    floating_point_type* y, int incy);

// level 3 matrix and matrix
// matrix matrix multiply
template<typename floating_point_type>
void cublas_gemm(
    cublasHandle_t handle, cublasOperation_t cutrans_a,
    cublasOperation_t cutrans_b, int m, int n, int k,
    const floating_point_type* alpha, const floating_point_type* a, int lda,
    const floating_point_type* b, int ldb,
    const floating_point_type* beta, floating_point_type* c, int ldc);

} // namespace oneflow

#endif // ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_
