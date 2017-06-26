#ifndef ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_
#define ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

template<typename floating_point_type>
void cublas_gemm(
    const cublasHandle_t& cublas_handle, cublasOperation_t cuTransA,
    cublasOperation_t cuTransB, int M, int N, int K,
    const floating_point_type alpha, const floating_point_type* A,
    int lda, const floating_point_type* B, int ldb,
    const floating_point_type beta, floating_point_type* C, int ldc);

template<typename floating_point_type>
void cublas_axpy(
    cublasHandle_t handle, int n,
    const floating_point_type *alpha,
    const floating_point_type *x, int incx,
    floating_point_type *y, int incy);

} // namespace oneflow

#endif // ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_
