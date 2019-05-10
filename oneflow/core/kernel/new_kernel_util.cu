#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

#define HALF_CHECK_FAILED                   \
  printf("use half need nvcc arch >= 530"); \
  assert(false)

__inline__ __device__ half hone() { return __float2half(1.0); }
__inline__ __device__ half hzero() { return __float2half(0.0); }

template<typename T>
__global__ void ReluForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : 0; }
}

template<typename T>
__global__ void ReluBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = y[i] > 0 ? dy[i] : 0; }
}

__inline__ half float16_2half(float16 x) {
  // TODO: Potential loss of accuracy
  half* ret = reinterpret_cast<half*>(&x);
  return *ret;
}

__inline__ float16 half2float16(half x) {
  // TODO: Potential loss of accuracy
  float16* ret = reinterpret_cast<float16*>(&x);
  return *ret;
}

__global__ void ReluForwardGpuHalf(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (__hgt(x[i], hzero())) {
      y[i] = x[i];
    } else {
      y[i] = hzero();
    }
  }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

__global__ void ReluBackwardGpuHalf(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half zero = __float2half(0.0);
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (__hgt(y[i], zero)) {
      dx[i] = dy[i];
    } else {
      dx[i] = zero;
    }
  }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void SigmoidForwardGpuHalf(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = __hdiv(hone(), __hadd(hone(), hexp(__hneg(x[i])))); }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

__global__ void SigmoidBackwardGpuHalf(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = __hmul(dy[i], __hmul(y[i], __hsub(hone(), y[i]))); }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

__global__ void TanHForwardGpuHalf(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    half ex = hexp(x[i]);
    half e_x = hexp(__hneg(x[i]));
    y[i] = __hdiv(__hsub(ex, e_x), __hadd(ex, e_x));
  }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void TanHBackwardGpuHalf(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = __hmul(dy[i], __hsub(hone(), __hmul(y[i], y[i]))); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

template<typename T>
__global__ void SigmoidForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 1.0 / (1.0 + std::exp(-x[i])); }
}

template<typename T>
__global__ void SigmoidBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dy[i] * y[i] * (1.0 - y[i]); }
}

template<typename T>
__global__ void TanHForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::tanh(x[i]); }
}

template<typename T>
__global__ void TanHBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dy[i] * (1.0 - y[i] * y[i]); }
}

__global__ void AxpyHalfGpu(const int n, const half alpha, const half* x, const int incx, half* y,
                            const int incy) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { y[i * incy] = __hfma(alpha, x[i * incx], y[i * incy]); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

cublasOperation_t CblasTrans2CublasTrans(CBLAS_TRANSPOSE trans) {
  cublasOperation_t cublas_trans;
  if (trans == CBLAS_TRANSPOSE::CblasNoTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_N;
  } else if (trans == CBLAS_TRANSPOSE::CblasTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_T;
  } else if (trans == CBLAS_TRANSPOSE::CblasConjTrans) {
    cublas_trans = cublasOperation_t::CUBLAS_OP_C;
  } else {
    // do nothing
  }
  return cublas_trans;
}

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
                 enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k, const T alpha,
                 const T* a, const T* b, const T beta, T* c) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);

  cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha, b, ldb,
                 a, lda, &beta, c, ldc);
}

static void HGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
                  enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                  const float16 alpha, const float16* a, const float16* b, const float16 beta,
                  float16* c) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  CudaCheck(cublasHgemm(ctx->cublas_tensor_op_math_handle(), cublas_trans_b, cublas_trans_a, n, m,
                        k, reinterpret_cast<const half*>(&alpha), reinterpret_cast<const half*>(b),
                        ldb, reinterpret_cast<const half*>(a), lda,
                        reinterpret_cast<const half*>(&beta), reinterpret_cast<half*>(c), ldc));
}

template<typename T>
static void BlobGemmImpl(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                         T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
  const int m = c->shape().At(0);
  const int n = c->shape().Count(1);
  const int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

  NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(),
                                          b->dptr<T>(), beta, c->mut_dptr<T>());
}

}  // namespace

#define GPU_KU_METHOD void NewKernelUtil<DeviceType::kGPU>::

GPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       float alpha, float beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<float>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
GPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       double alpha, double beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<double>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
GPU_KU_METHOD BlobGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                       float16 alpha, float16 beta, const Blob* a, const Blob* b, Blob* c) {
  BlobGemmImpl<float16>(ctx, trans_a, trans_b, alpha, beta, a, b, c);
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float alpha, const float* a,
                     const float* b, const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const double alpha, const double* a,
                     const double* b, const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float16 alpha, const float16* a,
                     const float16* b, const float16 beta, float16* c) {
  HGemm(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

GPU_KU_METHOD Relu(DeviceCtx* ctx, const int64_t n, const float* x, float* y) {
  ReluForwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

GPU_KU_METHOD Relu(DeviceCtx* ctx, const int64_t n, const double* x, double* y) {
  ReluForwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}
GPU_KU_METHOD Relu(DeviceCtx* ctx, const int64_t n, const float16* x, float16* y) {
  ReluForwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
}

GPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                           const float* dy, float* dx) {
  ReluBackwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

GPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                           const double* dy, double* dx) {
  ReluBackwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

GPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                           const float16* dy, float16* dx) {
  ReluBackwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy),
      reinterpret_cast<half*>(dx));
}

GPU_KU_METHOD Sigmoid(DeviceCtx* ctx, int64_t n, const float* x, float* y) {
  SigmoidForwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

GPU_KU_METHOD Sigmoid(DeviceCtx* ctx, int64_t n, const double* x, double* y) {
  SigmoidForwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

GPU_KU_METHOD Sigmoid(DeviceCtx* ctx, int64_t n, const float16* x, float16* y) {
  SigmoidForwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                          ctx->cuda_stream()>>>(n, reinterpret_cast<const half*>(x),
                                                reinterpret_cast<half*>(y));
}

GPU_KU_METHOD SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                              const float* dy, float* dx) {
  SigmoidBackwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

GPU_KU_METHOD SigmoidBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                              const double* dy, double* dx) {
  SigmoidBackwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

GPU_KU_METHOD SigmoidBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                              const float16* dy, float16* dx) {
  SigmoidBackwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                           ctx->cuda_stream()>>>(n, reinterpret_cast<const half*>(y),
                                                 reinterpret_cast<const half*>(dy),
                                                 reinterpret_cast<half*>(dx));
}

GPU_KU_METHOD TanH(DeviceCtx* ctx, int64_t n, const float* x, float* y) {
  TanHForwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

GPU_KU_METHOD TanH(DeviceCtx* ctx, int64_t n, const double* x, double* y) {
  TanHForwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
}

GPU_KU_METHOD TanH(DeviceCtx* ctx, int64_t n, const float16* x, float16* y) {
  TanHForwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
}

GPU_KU_METHOD TanHBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y,
                           const float* dy, float* dx) {
  TanHBackwardGpu<float>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

GPU_KU_METHOD TanHBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y,
                           const double* dy, double* dx) {
  TanHBackwardGpu<double>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

GPU_KU_METHOD TanHBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y,
                           const float16* dy, float16* dx) {
  TanHBackwardGpuHalf<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy),
      reinterpret_cast<half*>(dx));
}

GPU_KU_METHOD Axpy(DeviceCtx* ctx, const int n, const float alpha, const float* x, const int incx,
                   float* y, const int incy) {
  cublas_axpy<float>(ctx->cublas_pmh_handle(), n, &alpha, x, incx, y, incy);
}

GPU_KU_METHOD Axpy(DeviceCtx* ctx, const int n, const double alpha, const double* x, const int incx,
                   double* y, const int incy) {
  cublas_axpy<double>(ctx->cublas_pmh_handle(), n, &alpha, x, incx, y, incy);
}

GPU_KU_METHOD Axpy(DeviceCtx* ctx, const int n, const float16 alpha, const float16* x,
                   const int incx, float16* y, const int incy) {
  AxpyHalfGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, float16_2half(alpha), reinterpret_cast<const half*>(x), incx, reinterpret_cast<half*>(y),
      incy);
}

}  // namespace oneflow
