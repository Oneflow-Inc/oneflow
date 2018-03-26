#ifndef ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_
#define ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_

#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA

// level 1 vector and vector
// dot product
template<typename T>
void cublas_dot(cublasHandle_t handle, int n, const T* x, int incx, const T* y,
                int incy, T* result);

// swap x and y
template<typename T>
void cublas_swap(cublasHandle_t handle, int n, T* x, int incx, T* y, int incy);

// copy x into y
template<typename T>
void cublas_copy(cublasHandle_t handle, int n, const T* x, int incx, T* y,
                 int incy);

// y = a*x + y
template<typename T>
void cublas_axpy(cublasHandle_t handle, int n, const T* alpha, const T* x,
                 int incx, T* y, int incy);

// x = a*x
template<typename T>
void cublas_scal(cublasHandle_t handle, int n, const T* alpha, T* x, int incx);

// level 2 matrix and vector
// matrix vector multiply
template<typename T>
void cublas_gemv(cublasHandle_t handle, cublasOperation_t trans, int m, int n,
                 const T* alpha, const T* a, int lda, const T* x, int incx,
                 const T* beta, T* y, int incy);

// level 3 matrix and matrix
// matrix matrix multiply
template<typename T>
void cublas_gemm(cublasHandle_t handle, cublasOperation_t cutrans_a,
                 cublasOperation_t cutrans_b, int m, int n, int k,
                 const T* alpha, const T* a, int lda, const T* b, int ldb,
                 const T* beta, T* c, int ldc);

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_BLAS_CUBLAS_TEMPLATE_H_
