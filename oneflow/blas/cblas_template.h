#ifndef ONEFLOW_BLAS_CBLAS_TEMPLATE_H_
#define ONEFLOW_BLAS_CBLAS_TEMPLATE_H_

extern "C" {
#include "oneflow/blas/cblas.h"
}

namespace oneflow {

template<typename floating_point_type>
void cblas_gemm(
    const enum CBLAS_TRANSPOSE TransA, const enum CBLAS_TRANSPOSE TransB,
    const int M, const int N, const int K, const floating_point_type alpha,
    const floating_point_type* A, const int lda,
    const floating_point_type* B, const int ldb, const floating_point_type beta,
    floating_point_type* C, const int ldc) {
}

template<typename floating_point_type>
void cblas_axpy(
    const int N,
    const floating_point_type alpha,
    const floating_point_type *x, const int incx,
    floating_point_type *y, const int incy) {
  LOG(FATAL) << "floating_point_type should be float or double";
}

} // namespace oneflow

#endif // ONEFLOW_BLAS_CBLAS_TEMPLATE_H_
