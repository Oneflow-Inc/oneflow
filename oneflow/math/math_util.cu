#include <math_functions.h>  // CUDA's, not caffe's, for fabs, signbit

#include <cuda.h>
#include <cmath>
#include <cstdlib>
#include <cstring>

#include "math/math_util.h"
#include "device/device_alternate.h"

namespace caffe {

template <>
void caffe_gpu_gemm<float>(
    cublasHandle_t cublas_handle,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const float alpha, const float* A, const float* B, const float beta,
    float* C, cudaStream_t stream) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasSgemm(cublas_handle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}

template <>
void caffe_gpu_gemm<double>(
    cublasHandle_t cublas_handle,
    const CBLAS_TRANSPOSE TransA,
    const CBLAS_TRANSPOSE TransB, const int M, const int N, const int K,
    const double alpha, const double* A, const double* B, const double beta,
    double* C, cudaStream_t stream) {
  // Note that cublas follows fortran order.
  int lda = (TransA == CblasNoTrans) ? K : M;
  int ldb = (TransB == CblasNoTrans) ? N : K;
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  cublasOperation_t cuTransB =
      (TransB == CblasNoTrans) ? CUBLAS_OP_N : CUBLAS_OP_T;
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasDgemm(cublas_handle, cuTransB, cuTransA,
      N, M, K, &alpha, B, ldb, A, lda, &beta, C, N));
}
template <>
void caffe_gpu_gemv<float>(
    cublasHandle_t cublas_handle,
    const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const float alpha, const float* A, const float* x,
    const float beta, float* y, cudaStream_t stream) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasSgemv(cublas_handle, cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_gemv<double>(
    cublasHandle_t cublas_handle,
    const CBLAS_TRANSPOSE TransA, const int M,
    const int N, const double alpha, const double* A, const double* x,
    const double beta, double* y, cudaStream_t stream) {
  cublasOperation_t cuTransA =
      (TransA == CblasNoTrans) ? CUBLAS_OP_T : CUBLAS_OP_N;
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasDgemv(cublas_handle, cuTransA, N, M, &alpha,
      A, N, x, 1, &beta, y, 1));
}

template <>
void caffe_gpu_axpy<float>(
    cublasHandle_t cublas_handle,
    const int N, const float alpha, const float* X,
    float* Y, cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasSaxpy(cublas_handle, N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_axpy<double>(
    cublasHandle_t cublas_handle,
    const int N, const double alpha, const double* X,
    double* Y, cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasDaxpy(cublas_handle, N, &alpha, X, 1, Y, 1));
}

template <>
void caffe_gpu_scal<float>(
  cublasHandle_t cublas_handle,
  const int N, const float alpha, float *X, cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasSscal(cublas_handle, N, &alpha, X, 1));
}

template <>
void caffe_gpu_scal<double>(
  cublasHandle_t cublas_handle,
  const int N, const double alpha, double *X, cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasDscal(cublas_handle, N, &alpha, X, 1));
}

template <>
void caffe_gpu_axpby<float>(
  cublasHandle_t cublas_handle,
  const int N, const float alpha, const float* X,
    const float beta, float* Y, cudaStream_t stream) {
  caffe_gpu_scal<float>(cublas_handle, N, beta, Y, stream);
  caffe_gpu_axpy<float>(cublas_handle, N, alpha, X, Y, stream);
}

template <>
void caffe_gpu_axpby<double>(
  cublasHandle_t cublas_handle,
  const int N, const double alpha, const double* X,
    const double beta, double* Y, cudaStream_t stream) {
  caffe_gpu_scal<double>(cublas_handle, N, beta, Y, stream);
  caffe_gpu_axpy<double>(cublas_handle, N, alpha, X, Y, stream);
}

template <>
void caffe_gpu_dot<float>(
  cublasHandle_t cublas_handle,
  const int n, const float* x, const float* y,
    float* out, cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasSdot(cublas_handle, n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_dot<double>(
  cublasHandle_t cublas_handle,
  const int n, const double* x, const double* y,
    double * out, cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasDdot(cublas_handle, n, x, 1, y, 1, out));
}

template <>
void caffe_gpu_asum<float>(
  cublasHandle_t cublas_handle,
  const int n, const float* x, float* y, cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasSasum(cublas_handle, n, x, 1, y));
}

template <>
void caffe_gpu_asum<double>(
  cublasHandle_t cublas_handle,
  const int n, const double* x, double* y, cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasDasum(cublas_handle, n, x, 1, y));
}

template <>
void caffe_gpu_scale<float>(
  cublasHandle_t cublas_handle,
  const int n, const float alpha, const float *x, float* y,
  cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasScopy(cublas_handle, n, x, 1, y, 1));
  CUBLAS_CHECK(cublasSscal(cublas_handle, n, &alpha, y, 1));
}

template <>
void caffe_gpu_scale<double>(
  cublasHandle_t cublas_handle,
  const int n, const double alpha, const double *x, double* y,
  cudaStream_t stream) {
  if (stream) {
    CUBLAS_CHECK(cublasSetStream(cublas_handle, stream));
  }
  CUBLAS_CHECK(cublasDcopy(cublas_handle, n, x, 1, y, 1));
  CUBLAS_CHECK(cublasDscal(cublas_handle, n, &alpha, y, 1));
}

template <typename Dtype>
__global__ void set_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = alpha;
  }
}

template <typename Dtype>
void caffe_gpu_async_set(const int N, const Dtype alpha, Dtype* Y,
    cudaStream_t stream) {
  CHECK_NOTNULL(stream);
  if (alpha == 0) {
    CUDA_CHECK(cudaMemsetAsync(Y, 0, sizeof(Dtype) * N, stream));
    return;
  }
  set_kernel<Dtype><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, alpha, Y);
}

template void caffe_gpu_async_set<int>(const int N, const int alpha, int* Y,
    cudaStream_t stream);
template void caffe_gpu_async_set<float>(
    const int N, const float alpha, float* Y, cudaStream_t stream);
template void caffe_gpu_async_set<double>(
    const int N, const double alpha, double* Y, cudaStream_t stream);

template <typename Dtype>
__global__ void add_scalar_kernel(const int n, const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] += alpha;
  }
}

template <typename Dtype>
void caffe_gpu_async_copy(const int N, const Dtype* X, Dtype* Y,
  cudaStream_t stream) {
  if (X != Y) {
  // NOLINT_NEXT_LINE(caffe/alt_fn)
  CUDA_CHECK(cudaMemcpyAsync(Y, X, sizeof(Dtype)* N, cudaMemcpyDefault,
    stream));
  }
}

template void caffe_gpu_async_copy<int>(const int N, const int* X, int* Y,
  cudaStream_t stream);
template void caffe_gpu_async_copy<unsigned int>(const int N,
  const unsigned int* X, unsigned int* Y, cudaStream_t stream);
template void caffe_gpu_async_copy<float>(const int N, const float* X,
  float* Y, cudaStream_t stream);
template void caffe_gpu_async_copy<double>(const int N, const double* X,
  double* Y, cudaStream_t stream);

template <>
void caffe_gpu_add_scalar(const int N, const float alpha, float* Y,
  cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, alpha, Y);
}

template <>
void caffe_gpu_add_scalar(const int N, const double alpha, double* Y,
  cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_scalar_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, alpha, Y);
}

template <typename Dtype>
__global__ void add_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] + b[index];
  }
}

template <>
void caffe_gpu_add<float>(const int N, const float* a, const float* b,
    float* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_add<double>(const int N, const double* a, const double* b,
    double* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  add_kernel<double> <<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void sub_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] - b[index];
  }
}

template <>
void caffe_gpu_sub<float>(const int N, const float* a, const float* b,
    float* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_sub<double>(const int N, const double* a, const double* b,
    double* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  sub_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void mul_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] * b[index];
  }
}

template <>
void caffe_gpu_mul<float>(const int N, const float* a,
    const float* b, float* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_mul<double>(const int N, const double* a,
    const double* b, double* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  mul_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void div_kernel(const int n, const Dtype* a,
    const Dtype* b, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = a[index] / b[index];
  }
}

template <>
void caffe_gpu_div<float>(const int N, const float* a,
    const float* b, float* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, b, y);
}

template <>
void caffe_gpu_div<double>(const int N, const double* a,
    const double* b, double* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  div_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, b, y);
}

template <typename Dtype>
__global__ void abs_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = abs(a[index]);
  }
}

template <>
void caffe_gpu_abs<float>(const int N, const float* a, float* y,
  cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, y);
}

template <>
void caffe_gpu_abs<double>(const int N, const double* a, double* y,
  cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  abs_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, y);
}


template <typename Dtype>
__global__ void exp_kernel(const int n, const Dtype* a, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = exp(a[index]);
  }
}

template <>
void caffe_gpu_exp<float>(const int N, const float* a, float* y,
  cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, y);
}

template <>
void caffe_gpu_exp<double>(const int N, const double* a, double* y,
  cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  exp_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, y);
}

template <typename Dtype>
__global__ void powx_kernel(const int n, const Dtype* a,
    const Dtype alpha, Dtype* y) {
  CUDA_KERNEL_LOOP(index, n) {
    y[index] = pow(a[index], alpha);
  }
}

template <>
void caffe_gpu_powx<float>(const int N, const float* a,
    const float alpha, float* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<float><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, alpha, y);
}

template <>
void caffe_gpu_powx<double>(const int N, const double* a,
    const double alpha, double* y, cudaStream_t stream) {
  // NOLINT_NEXT_LINE(whitespace/operators)
  powx_kernel<double><<<CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0, stream>>>(
      N, a, alpha, y);
}

//TODO(xcdu):caffe_gpu_set need to be refined.
template <typename Dtype>
void caffe_gpu_set(const int N, const Dtype alpha, Dtype* Y, cudaStream_t stream) {
  if (alpha == 0) {
    CUDA_CHECK(cudaMemset(Y, 0, sizeof(Dtype)* N));  // NOLINT(caffe/alt_fn)
    return;
  }
  // NOLINT_NEXT_LINE(whitespace/operators)
  set_kernel<Dtype> << <CAFFE_GET_BLOCKS(N), CAFFE_CUDA_NUM_THREADS, 0 ,stream>> >(
    N, alpha, Y);
}

template void caffe_gpu_set<int>(const int N, const int alpha, int* Y, cudaStream_t stream);
template void caffe_gpu_set<float>(const int N, const float alpha, float* Y, cudaStream_t stream);
template void caffe_gpu_set<double>(const int N, const double alpha, double* Y, cudaStream_t stream);


DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sign, y[index] = (Dtype(0) < x[index]);
                                      - (x[index] < Dtype(0)))
DEFINE_AND_INSTANTIATE_GPU_UNARY_FUNC(sgnbit, y[index] = signbit(x[index]));

}  // namespace caffe
