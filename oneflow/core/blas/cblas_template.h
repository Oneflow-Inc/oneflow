#ifndef ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_
#define ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_

extern "C" {
#include "oneflow/core/blas/cblas.h"
}

namespace oneflow {

// level 1 vector and vector
// dot product
template<typename floating_point_type>
void cblas_dot(
    const int N, 
    const floating_point_type* X, const int incX,
    const floating_point_type* Y, const int incY);

// swap x and y
template<typename floating_point_type>
void cblas_swap(
    const int N,
    floating_point_type* X, const int incX,
    floating_point_type* Y, const int incY);

// copy x into y
template<typename floating_point_type>
void cblas_copy(
    const int N,
    const floating_point_type* X, const int incX,
    floating_point_type* Y, const int incY);

// y = a*x + y
template<typename floating_point_type>
void cblas_axpy(
    const int N,
    const floating_point_type alpha,
    const floating_point_type* X, const int incX,
    floating_point_type* Y, const int incY);

// x = a*x
template<typename floating_point_type>
void cblas_scal(
    const int N,
    const floating_point_type alpha,
    floating_point_type* X, const int incX);

// level 2 matrix and vector
// matrix vector multiply
template<typename floating_point_type>
void cblas_gemv(
    const enum CBLAS_ORDER order,const enum CBLAS_TRANSPOSE TransA,
    const int M, const int N, const floating_point_type alpha,
    const floating_point_type* A, const int lda,
    const floating_point_type* X, const int incX,
    const floating_point_type beta, floating_point_type* Y, const int incY);

// level 3 matrix and matrix
// matrix matrix multiply
template<typename floating_point_type>
void cblas_gemm(
    const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE TransA,
    const enum CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const floating_point_type alpha, const floating_point_type* A,
    const int lda, const floating_point_type* B, const int ldb,
    const floating_point_type beta, floating_point_type* C, const int ldc);

} // namespace oneflow

#endif // ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_
