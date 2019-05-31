#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

// Type Trait: IsHalf
template<typename T>
struct IsHalf : std::integral_constant<bool, false> {};

template<>
struct IsHalf<half> : std::integral_constant<bool, true> {};

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

template<>
__global__ void ReluForwardGpu<half>(const int n, const half* x, half* y) {
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

template<typename T>
__global__ void ReluBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = y[i] > 0 ? dy[i] : 0; }
}

template<>
__global__ void ReluBackwardGpu<half>(const int n, const half* y, const half* dy, half* dx) {
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

template<typename T>
__global__ void SigmoidForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = 1.0 / (1.0 + std::exp(-x[i])); }
}

template<>
__global__ void SigmoidForwardGpu<half>(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = __hdiv(hone(), __hadd(hone(), hexp(__hneg(x[i])))); }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

template<typename T>
__global__ void SigmoidBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dy[i] * y[i] * (1.0 - y[i]); }
}

template<>
__global__ void SigmoidBackwardGpu<half>(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = __hmul(dy[i], __hmul(y[i], __hsub(hone(), y[i]))); }
#else
  HALF_CHECK_FAILED;
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

template<typename T>
__global__ void TanHForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::tanh(x[i]); }
}

template<>
__global__ void TanHForwardGpu<half>(const int n, const half* x, half* y) {
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

template<typename T>
__global__ void TanHBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = dy[i] * (1.0 - y[i] * y[i]); }
}

template<>
__global__ void TanHBackwardGpu<half>(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = __hmul(dy[i], __hsub(hone(), __hmul(y[i], y[i]))); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void AxpyHalfGpu(const int n, const half alpha, const half* x, const int incx, half* y,
                            const int incy) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { y[i * incy] = __hfma(alpha, x[i * incx], y[i * incy]); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
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

std::tuple<int, int, int, cublasOperation_t, cublasOperation_t> PrepareToCallCublasGemm(
    enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b, const int m, const int n,
    const int k) {
  int lda = (trans_a == CblasNoTrans) ? k : m;
  int ldb = (trans_b == CblasNoTrans) ? n : k;
  int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  return std::make_tuple(lda, ldb, ldc, cublas_trans_a, cublas_trans_b);
}

template<typename T>
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
                 enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                 const T* alpha, const T* a, const T* b, const T* beta, T* c) {
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallCublasGemm(trans_a, trans_b, m, n, k);

  cublasHandle_t handle;
  if (IsHalf<T>::value) {
    handle = ctx->cublas_tensor_op_math_handle();
  } else {
    handle = ctx->cublas_pmh_handle();
  }
  cublas_gemm<T>(handle, cublas_trans_b, cublas_trans_a, n, m, k, alpha, b, ldb, a, lda, beta, c,
                 ldc);
}

void HGemmWithFloat(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a,
                    enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                    const float* alpha, const half* a, const half* b, const float* beta, half* c) {
  int lda, ldb, ldc;
  cublasOperation_t cublas_trans_a, cublas_trans_b;
  std::tie(lda, ldb, ldc, cublas_trans_a, cublas_trans_b) =
      PrepareToCallCublasGemm(trans_a, trans_b, m, n, k);

  cudaDataType_t data_type = GetCudaDataType(DataType::kFloat16);
  CudaCheck(cublasSgemmEx(ctx->cublas_tensor_op_math_handle(), cublas_trans_b, cublas_trans_a, n, m,
                          k, alpha, b, data_type, ldb, a, data_type, lda, beta, c, data_type, ldc));
}

std::tuple<int, int, int> CalcMNKForGemm(enum CBLAS_TRANSPOSE trans_a, const Blob* a,
                                         const Blob* c) {
  int m = c->shape().At(0);
  int n = c->shape().Count(1);
  int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);
  return std::make_tuple(m, n, k);
}

template<typename T>
static void BlobGemmImpl(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                         T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
  int m, n, k;
  std::tie(m, n, k) = CalcMNKForGemm(trans_a, a, c);
  NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(),
                                          b->dptr<T>(), beta, c->mut_dptr<T>());
}

template<typename T>
__global__ void AssignStridedAddrGpu(T** dev_ptrs, T* start_ptr, int32_t stride_len,
                                     int32_t stride_num) {
  CUDA_1D_KERNEL_LOOP(i, stride_num) { dev_ptrs[i] = start_ptr + i * stride_len; }
}

template<typename T>
void AssignStridedAddr(DeviceCtx* ctx, T** dev_ptrs, T* start_ptr, int stride_len, int stride_num) {
  AssignStridedAddrGpu<T>
      <<<BlocksNum4ThreadsNum(stride_num), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          dev_ptrs, start_ptr, stride_len, stride_num);
}

template<typename T>
static void BatchedGemmImpl(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                            const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                            int batch_size, int m, int n, int k, const T alpha, const T* a,
                            const T* b, const T beta, T* c, T** buf) {
  const int a_stride = m * k;
  const int b_stride = k * n;
  const int c_stride = m * n;
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  T** dev_a_ptrs = buf;
  T** dev_b_ptrs = buf + batch_size;
  T** dev_c_ptrs = buf + 2 * batch_size;
  AssignStridedAddr<T>(ctx, dev_a_ptrs, const_cast<T*>(a), a_stride, batch_size);
  AssignStridedAddr<T>(ctx, dev_b_ptrs, const_cast<T*>(b), b_stride, batch_size);
  AssignStridedAddr<T>(ctx, dev_c_ptrs, c, c_stride, batch_size);
#if CUDA_VERSION >= 9010
  cudaDataType_t data_type = CudaDataType<T>::value;
  cublasGemmBatchedEx(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
                      reinterpret_cast<const void*>(&alpha),
                      reinterpret_cast<const void**>(const_cast<const T**>(dev_b_ptrs)), data_type,
                      ldb, reinterpret_cast<const void**>(const_cast<const T**>(dev_a_ptrs)),
                      data_type, lda, reinterpret_cast<const void*>(&beta),
                      reinterpret_cast<void**>(dev_c_ptrs), data_type, ldc, batch_size, data_type,
                      CUBLAS_GEMM_DEFAULT);
#else
  cublas_gemmBatched<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha,
                        const_cast<const T**>(dev_b_ptrs), ldb, const_cast<const T**>(dev_a_ptrs),
                        lda, &beta, dev_c_ptrs, ldc, batch_size);
#endif
}

template<int32_t NDIMS>
struct Int32Array {
  int32_t val[NDIMS];
};

template<int32_t NDIMS>
__device__ int32_t GetXIndex(const int32_t* y_shape, const int32_t* x_strides, int32_t y_idx) {
  int32_t x_idx = 0;
  for (int32_t i = NDIMS - 1; i >= 0; --i) {
    x_idx += (y_idx % y_shape[i]) * x_strides[i];
    y_idx /= y_shape[i];
  }
  return x_idx;
}

template<int32_t NDIMS, typename T>
__global__ void TransposeGpu(const Int32Array<NDIMS> y_shape, const Int32Array<NDIMS> x_strides,
                             const int32_t elem_cnt, const T* x, T* y) {
  __shared__ int32_t x_strides_shared[NDIMS];
  __shared__ int32_t y_dims_shared[NDIMS];
  const int32_t tid = threadIdx.x;
  if (tid < NDIMS) {
    y_dims_shared[tid] = y_shape.val[tid];
    x_strides_shared[tid] = x_strides.val[tid];
  }
  __syncthreads();
  CUDA_1D_KERNEL_LOOP(y_idx, elem_cnt) {
    const int32_t x_idx = GetXIndex<NDIMS>(y_dims_shared, x_strides_shared, y_idx);
#if __CUDA_ARCH__ >= 350
    y[y_idx] = __ldg(x + x_idx);
#else
    y[y_idx] = x[x_idx];
#endif
  }
}

template<int32_t NDIMS, typename T>
void TransposeImpl(DeviceCtx* ctx, const Shape& x_shape, const Shape& y_shape,
                   const PbRf<int32_t>& permutation, const int64_t elem_cnt, const T* x, T* y) {
  CHECK_LE(y_shape.elem_cnt(), MaxVal<int32_t>::value);
  Int32Array<NDIMS> y_shape_struct;
  FOR_RANGE(int32_t, i, 0, NDIMS) { y_shape_struct.val[i] = y_shape.At(i); }
  Int32Array<NDIMS> x_strides;
  int32_t buff[NDIMS];
  int32_t cur_stride = 1;
  for (int32_t i = NDIMS - 1; i >= 0; --i) {
    buff[i] = cur_stride;
    cur_stride *= x_shape.At(i);
  }
  for (int32_t i = 0; i < NDIMS; ++i) { x_strides.val[i] = buff[permutation[i]]; }
  TransposeGpu<NDIMS, T>
      <<<SMBlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          y_shape_struct, x_strides, elem_cnt, x, y);
}

template<typename T>
struct TransposeUtil final {
#define MAKE_TRANSPOSE_SWITCH_ENTRY(func_name, NDIMS) func_name<NDIMS, T>
  DEFINE_STATIC_SWITCH_FUNC(void, TransposeImpl, MAKE_TRANSPOSE_SWITCH_ENTRY,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
};

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
GPU_KU_METHOD BlobHGemmWithFloat(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                                 enum CBLAS_TRANSPOSE trans_b, float alpha, float beta,
                                 const Blob* a, const Blob* b, Blob* c) {
  int m, n, k;
  std::tie(m, n, k) = CalcMNKForGemm(trans_a, a, c);
  NewKernelUtil<DeviceType::kGPU>::OFHGemmWithFloat(ctx, trans_a, trans_b, m, n, k, alpha,
                                                    a->dptr<float16>(), b->dptr<float16>(), beta,
                                                    c->mut_dptr<float16>());
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float alpha, const float* a,
                     const float* b, const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, &alpha, a, b, &beta, c);
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const double alpha, const double* a,
                     const double* b, const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, &alpha, a, b, &beta, c);
}
GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const float16 alpha, const float16* a,
                     const float16* b, const float16 beta, float16* c) {
  Gemm<half>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, reinterpret_cast<const half*>(&alpha),
             reinterpret_cast<const half*>(a), reinterpret_cast<const half*>(b),
             reinterpret_cast<const half*>(&beta), reinterpret_cast<half*>(c));
}
GPU_KU_METHOD OFHGemmWithFloat(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                               enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                               const float alpha, const float16* a, const float16* b,
                               const float beta, float16* c) {
  HGemmWithFloat(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, &alpha,
                 reinterpret_cast<const half*>(a), reinterpret_cast<const half*>(b), &beta,
                 reinterpret_cast<half*>(c));
}

GPU_KU_METHOD OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                            enum CBLAS_TRANSPOSE trans_b, const int batch_size, const int m,
                            const int n, const int k, const float alpha, const float* a,
                            const float* b, const float beta, float* c, float** buf) {
  BatchedGemmImpl<float>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                         beta, c, buf);
}
GPU_KU_METHOD OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                            enum CBLAS_TRANSPOSE trans_b, const int batch_size, const int m,
                            const int n, const int k, const double alpha, const double* a,
                            const double* b, const double beta, double* c, double** buf) {
  BatchedGemmImpl<double>(ctx, CblasRowMajor, trans_a, trans_b, batch_size, m, n, k, alpha, a, b,
                          beta, c, buf);
}
GPU_KU_METHOD OFBatchedGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a,
                            enum CBLAS_TRANSPOSE trans_b, const int batch_size, const int m,
                            const int n, const int k, const float16 alpha, const float16* a,
                            const float16* b, const float16 beta, float16* c, float16** buf) {
  // TODO(niuchong): implement half batched gemm
  UNIMPLEMENTED();
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
  ReluForwardGpu<half><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
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
  ReluBackwardGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
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
  SigmoidForwardGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
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
  SigmoidBackwardGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy),
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
  TanHForwardGpu<half><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
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
  TanHBackwardGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
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

#define TRANSPOSE_CHECK                                 \
  CHECK_LE(y_shape.elem_cnt(), MaxVal<int32_t>::value); \
  CHECK_EQ(num_axis, y_shape.NumAxes());                \
  CHECK_EQ(num_axis, x_shape.NumAxes())

GPU_KU_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const float* x, float* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<float>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                            permutation, elem_cnt, x, y);
}

GPU_KU_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const double* x, double* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<double>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                             permutation, elem_cnt, x, y);
}

GPU_KU_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const float16* x, float16* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<half>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape, permutation,
                                           elem_cnt, reinterpret_cast<const half*>(x),
                                           reinterpret_cast<half*>(y));
}

#undef TRANSPOSE_CHECK

template<typename T>
__device__ T MaxWithLogThreshold(T x) {
  const T threshold = 1e-20;
  return x > threshold ? x : threshold;
}

template __device__ float MaxWithLogThreshold(float x);
template __device__ double MaxWithLogThreshold(double x);

template<>
__device__ half MaxWithLogThreshold(half x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  half threshold = hexp2(__float2half(-14.0));
  if (__hgt(x, threshold)) { return x; }
  return threshold;
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

template<typename T>
__device__ T SafeLog(T x) {
  return logf(MaxWithLogThreshold(x));
}

template __device__ float SafeLog(float x);
template __device__ double SafeLog(double x);

template<>
__device__ half SafeLog(half x) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  hlog(MaxWithLogThreshold<half>(x));
#else
  printf("use half need nvcc arch >= 530");
  assert(false);
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)*/
}

}  // namespace oneflow
