#ifndef ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_
#define ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_

extern "C" {
#include "oneflow/core/blas/cblas.h"
}

namespace oneflow {

// level 1 vector and vector
// dot product
template<typename FloatingPointType>
FloatingPointType cblas_dot(const int n, const FloatingPointType* x,
                            const int incx, const FloatingPointType* y,
                            const int incy);

// swap x and y
template<typename FloatingPointType>
void cblas_swap(const int n, FloatingPointType* x, const int incx,
                FloatingPointType* y, const int incy);

// copy x into y
template<typename FloatingPointType>
void cblas_copy(const int n, const FloatingPointType* x, const int incx,
                FloatingPointType* y, const int incy);

// y = a*x + y
template<typename FloatingPointType>
void cblas_axpy(const int n, const FloatingPointType alpha,
                const FloatingPointType* x, const int incx,
                FloatingPointType* y, const int incy);

// x = a*x
template<typename FloatingPointType>
void cblas_scal(const int n, const FloatingPointType alpha,
                FloatingPointType* x, const int incx);

// level 2 matrix and vector
// matrix vector multiply
template<typename FloatingPointType>
void cblas_gemv(const enum CBLAS_ORDER order,
                const enum CBLAS_TRANSPOSE trans_a, const int m, const int n,
                const FloatingPointType alpha, const FloatingPointType* a,
                const int lda, const FloatingPointType* x, const int incx,
                const FloatingPointType beta, FloatingPointType* y,
                const int incy);

// level 3 matrix and matrix
// matrix matrix multiply
template<typename FloatingPointType>
void cblas_gemm(const enum CBLAS_ORDER order,
                const enum CBLAS_TRANSPOSE trans_a,
                const enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
                const int k, const FloatingPointType alpha,
                const FloatingPointType* a, const int lda,
                const FloatingPointType* b, const int ldb,
                const FloatingPointType beta, FloatingPointType* c,
                const int ldc);
}  // namespace oneflow

#endif  // ONEFLOW_CORE_BLAS_CBLAS_TEMPLATE_H_
