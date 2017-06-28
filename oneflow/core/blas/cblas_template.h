#ifndef ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_
#define ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_

extern "C" {
#include "oneflow/core/blas/cblas.h"
}

namespace oneflow {

// level 1 vector and vector
// dot product
template<typename floating_point_type>
floating_point_type cblas_dot(
    const int n,
    const floating_point_type* x, const int incx,
    const floating_point_type* y, const int incy);

// swap x and y
template<typename floating_point_type>
void cblas_swap(
    const int n,
    floating_point_type* x, const int incx,
    floating_point_type* y, const int incy);

// copy x into y
template<typename floating_point_type>
void cblas_copy(
    const int n,
    const floating_point_type* x, const int incx,
    floating_point_type* y, const int incy);

// y = a*x + y
template<typename floating_point_type>
void cblas_axpy(
    const int n,
    const floating_point_type alpha,
    const floating_point_type* x, const int incx,
    floating_point_type* y, const int incy);

// x = a*x
template<typename floating_point_type>
void cblas_scal(
    const int n,
    const floating_point_type alpha,
    floating_point_type* x, const int incx);

// level 2 matrix and vector
// matrix vector multiply
template<typename floating_point_type>
void cblas_gemv(
    const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
    const int m, const int n, const floating_point_type alpha,
    const floating_point_type* a, const int lda,
    const floating_point_type* x, const int incx,
    const floating_point_type beta, floating_point_type* y, const int incy);

// level 3 matrix and matrix
// matrix matrix multiply
template<typename floating_point_type>
void cblas_gemm(
    const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
    const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
    const floating_point_type alpha, const floating_point_type* a,
    const int lda, const floating_point_type* b, const int ldb,
    const floating_point_type beta, floating_point_type* c, const int ldc);
}  // namespace oneflow

#endif  // ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_
