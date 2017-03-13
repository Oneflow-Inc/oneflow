#ifndef CAFFE_UTIL_MATH_FUNCTIONS_H_
#define CAFFE_UTIL_MATH_FUNCTIONS_H_

#include <stdint.h>
#include <cmath>  // for std::fabs and std::signbit
#include <cstdlib> // for calloc
#include <glog/logging.h>
#include "device/device_alternate.h"
#include "math/mkl_alternate.h"
#include "common/rng.h"

namespace caffe {

// Caffe gemm provides a simpler interface to the gemm functions, with the
// limitation that the data has to be contiguous in memory.

// NOTE(jiyuan): make cuda-interface explicit to cublasHandle_t and cudaStream_t,
// so that these functions can be bound to different DeviceContext
template <typename Dtype>
void caffe_cpu_gemm(const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C);

template <typename Dtype>
void caffe_cpu_gemv(const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y);

template <typename Dtype>
void caffe_axpy(const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y);

template <typename Dtype>
void caffe_cpu_axpby(const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y);

template <typename Dtype>
void caffe_set(const int N, const Dtype alpha, Dtype *X);

inline void caffe_memset(const size_t N, const int alpha, void* X) {
  memset(X, alpha, N);  // NOLINT(caffe/alt_fn)
}

template <typename Dtype>
void caffe_add_scalar(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_scal(const int N, const Dtype alpha, Dtype *X);

template <typename Dtype>
void caffe_sqr(const int N, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_add(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_div(const int N, const Dtype* a, const Dtype* b, Dtype* y);

template <typename Dtype>
void caffe_powx(const int n, const Dtype* a, const Dtype b, Dtype* y);

template <typename Dtype>
void caffe_exp(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
void caffe_abs(const int n, const Dtype* a, Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_dot(const int n, const Dtype* x, const Dtype* y);

template <typename Dtype>
Dtype caffe_cpu_strided_dot(const int n, const Dtype* x, const int incx,
    const Dtype* y, const int incy);

template <typename Dtype>
int caffe_cpu_hamming_distance(const int n, const Dtype* x, const Dtype* y);

// Returns the sum of the absolute values of the elements of vector x
template <typename Dtype>
Dtype caffe_cpu_asum(const int n, const Dtype* x);

// the branchless, type-safe version from
// http://stackoverflow.com/questions/1903954/is-there-a-standard-sign-function-signum-sgn-in-c-c
template<typename Dtype>
inline int8_t caffe_sign(Dtype val) {
  return (Dtype(0) < val) - (val < Dtype(0));
}

// The following two macros are modifications of DEFINE_VSL_UNARY_FUNC
//   in include/caffe/util/mkl_alternate.hpp authored by @Rowland Depp.
// Please refer to commit 7e8ef25c7 of the boost-eigen branch.
// Git cherry picking that commit caused a conflict hard to resolve and
//   copying that file in convenient for code reviewing.
// So they have to be pasted here temporarily.
#define DEFINE_CAFFE_CPU_UNARY_FUNC(name, operation) \
  template<typename Dtype> \
  void caffe_cpu_##name(const int n, const Dtype* x, Dtype* y) { \
    CHECK_GT(n, 0); CHECK(x); CHECK(y); \
    for (int i = 0; i < n; ++i) { \
      operation; \
    } \
  }

// output is 1 for the positives, 0 for zero, and -1 for the negatives
DEFINE_CAFFE_CPU_UNARY_FUNC(sign, y[i] = caffe_sign<Dtype>(x[i]));

// This returns a nonzero value if the input has its sign bit set.
// The name sngbit is meant to avoid conflicts with std::signbit in the macro.
// The extra parens are needed because CUDA < 6.5 defines signbit as a macro,
// and we don't want that to expand here when CUDA headers are also included.
DEFINE_CAFFE_CPU_UNARY_FUNC(sgnbit, \
    y[i] = static_cast<bool>((std::signbit)(x[i])));

DEFINE_CAFFE_CPU_UNARY_FUNC(fabs, y[i] = std::fabs(x[i]));

template <typename Dtype>
void caffe_cpu_scale(const int n, const Dtype alpha, const Dtype *x, Dtype* y);
template <typename Dtype>
void caffe_rng_uniform(const int n, const Dtype minimum, const Dtype maximum,
  Dtype* dev);
template <typename Dtype>
void caffe_rng_discrete_uniform(const int n, const Dtype minimum,
  const Dtype maximum, Dtype* dev);
template <typename Dtype>
void caffe_rng_gaussian(const int n, const Dtype mean, const Dtype stddev,
  Dtype* dev);
template <typename Dtype>
void caffe_rng_bernoulli(const int n, const Dtype non_zero_probability,
  Dtype* dev);
template <typename Dtype>
void caffe_rng_positive_unitball(const int count,const int num, const int dim,
  Dtype* dev);

#ifndef CPU_ONLY  // GPU

// Decaf gpu gemm provides an interface that is almost the same as the cpu
// gemm function - following the c convention and calling the fortran-order
// gpu code under the hood.
template <typename Dtype>
void caffe_gpu_gemm(
    cublasHandle_t cublas_handle,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const Dtype alpha, const Dtype* A, const Dtype* B, const Dtype beta,
    Dtype* C, cudaStream_t stream = NULL);

template <typename Dtype>
void caffe_gpu_gemv(
    cublasHandle_t cublas_handle,
    const CBLAS_TRANSPOSE TransA, const int M, const int N,
    const Dtype alpha, const Dtype* A, const Dtype* x, const Dtype beta,
    Dtype* y, cudaStream_t stream = NULL);

template <typename Dtype>
void caffe_gpu_axpy(
    cublasHandle_t cublas_handle,
    const int N, const Dtype alpha, const Dtype* X,
    Dtype* Y, cudaStream_t stream = NULL);

template <typename Dtype>
void caffe_gpu_axpby(
    cublasHandle_t cublas_handle,
    const int N, const Dtype alpha, const Dtype* X,
    const Dtype beta, Dtype* Y, cudaStream_t stream = NULL);

template <typename Dtype>
void caffe_gpu_async_set(const int N, const Dtype alpha, Dtype *X,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_async_copy(const int N, const Dtype* X, Dtype* Y,
  cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_add_scalar(const int N, const Dtype alpha, Dtype *X,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_scal(
    cublasHandle_t cublas_handle,
    const int N, const Dtype alpha, Dtype *X,
    cudaStream_t stream = NULL);

//TODO(xcdu):2015.10.9 async
template <typename Dtype>
void caffe_copy(const int N, const Dtype *X, Dtype *Y);

template <typename Dtype>
void caffe_gpu_add(const int N, const Dtype* a, const Dtype* b, Dtype* y,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_sub(const int N, const Dtype* a, const Dtype* b, Dtype* y,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_mul(const int N, const Dtype* a, const Dtype* b, Dtype* y,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_div(const int N, const Dtype* a, const Dtype* b, Dtype* y,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_abs(const int n, const Dtype* a, Dtype* y,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_exp(const int n, const Dtype* a, Dtype* y,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_powx(const int n, const Dtype* a, const Dtype b, Dtype* y,
    cudaStream_t stream);

//TODO(xcdu):the location of caffe_gpu_set need to be refined;
template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_dot(
    cublasHandle_t cublas_handle,
    const int n, const Dtype* x, const Dtype* y, Dtype* out,
    cudaStream_t stream = NULL);

template <typename Dtype>
void caffe_gpu_asum(
    cublasHandle_t cublas_handle,
    const int n, const Dtype* x, Dtype* y,
    cudaStream_t stream = NULL);

template<typename Dtype>
void caffe_gpu_sign(const int n, const Dtype* x, Dtype* y,
    cudaStream_t stream);

template<typename Dtype>
void caffe_gpu_sgnbit(const int n, const Dtype* x, Dtype* y,
    cudaStream_t stream);

template <typename Dtype>
void caffe_gpu_scale(
    cublasHandle_t cublas_handle,
    const int n, const Dtype alpha, const Dtype *x, Dtype* y,
    cudaStream_t stream = NULL);

#define DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(name, operation) \
template<typename Dtype> \
__global__ void name##_kernel(const int n, const Dtype* x, Dtype* y) { \
  CUDA_KERNEL_LOOP(index, n) { \
    operation; \
  } \
} \
template <> \
void caffe_gpu_##name<float>(const int n, const float* x, float* y, cudaStream_t stream) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<float><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, stream>>>( \
      n, x, y); \
} \
template <> \
void caffe_gpu_##name<double>(const int n, const double* x, double* y, cudaStream_t stream) { \
  /* NOLINT_NEXT_LINE(whitespace/operators) */ \
  name##_kernel<double><<<CAFFE_GET_BLOCKS(n), CAFFE_CUDA_NUM_THREADS, 0, stream>>>( \
      n, x, y); \
}

#endif  // !CPU_ONLY

}  // namespace caffe

#endif  // CAFFE_UTIL_MATH_FUNCTIONS_H_
