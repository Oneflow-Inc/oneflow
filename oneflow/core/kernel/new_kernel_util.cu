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
static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  const int m, const int n, const int k, const T alpha, const T* a, const T* b,
  const T beta, T* c) {  
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);

  cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha, b,
  ldb, a, lda, &beta, c, ldc);
}

static void HGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
  const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
  const float16 beta, float16* c) {
  const int lda = (trans_a == CblasNoTrans) ? k : m;
  const int ldb = (trans_b == CblasNoTrans) ? n : k;
  const int ldc = n;

  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  CudaCheck(cublasHgemm(ctx->cublas_tensor_op_math_handle(), cublas_trans_b, cublas_trans_a, n, m,
        k, reinterpret_cast<const half*>(&alpha),
        reinterpret_cast<const half*>(b), ldb, reinterpret_cast<const half*>(a),
        lda, reinterpret_cast<const half*>(&beta), reinterpret_cast<half*>(c),
        ldc));
}

template<typename T>
static void BlobGemmImpl(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                      T alpha, T beta, const Blob* a, const Blob* b, Blob* c) {
  const int m = c->shape().At(0);
  const int n = c->shape().Count(1);
  const int k = (trans_a == CblasNoTrans) ? a->shape().Count(1) : a->shape().At(0);

  NewKernelUtil<DeviceType::kGPU>::OFGemm(ctx, trans_a, trans_b, m, n, k, alpha, a->dptr<T>(), b->dptr<T>(), beta,
           c->mut_dptr<T>());
}
template<typename T>
__device__ __forceinline__ T NewReduceCoreAdd(const T x, const T y) {
  return x + y;
}

template<typename T>
__device__ __forceinline__ T NewReduceCoreMax(const T x, const T y) {
  return x > y ? x : y;
}

template<>
__device__ __forceinline__ half NewReduceCoreAdd<half>(const half x, const half y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return __hadd(x, y);
#else
  HALF_CHECK_FAILED;
  return x;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

template<>
__device__ __forceinline__ half NewReduceCoreMax<half>(const half x, const half y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  return __hgt(x, y) ? x : y;
#else
  HALF_CHECK_FAILED;
  return x;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

template<typename T, T (*reduce_core_func)(const T, const T)>
__device__ void ShrinkColsGpu(const size_t row_num, const size_t thread_col_num, const T* x,
                                 const size_t x_col_num, const size_t x_lda, T* y,
                                 const size_t y_col_num, const size_t y_lda) {
  const size_t thread_num = blockDim.x * gridDim.x;
  const size_t total_shrink_scale = thread_col_num / y_col_num;
  CUDA_1D_KERNEL_LOOP(index, row_num * thread_col_num) {
    const int32_t thread_col = index % thread_col_num;
    if (((index / thread_num) % total_shrink_scale) != thread_col / y_col_num) { continue; }
    const int32_t row = index / thread_col_num;
    const int32_t col = thread_col % y_col_num;
    const int32_t x_start = row * x_lda + col;
    const int32_t x_end = row * x_lda + x_col_num;
    T reduced = x[x_start];
    for (int32_t x_index = x_start + y_col_num; x_index < x_end; x_index += y_col_num) {
      reduced = reduce_core_func(reduced, x[x_index]);
    }
    y[row * y_lda + col] = reduced;
  }
}

template<typename T, T (*reduce_core_func)(const T, const T), size_t shift_size = 2>
__global__ void RowReduceGpu(const size_t row_num, const size_t col_num, const T* x, T* y,
                                   T* temp_storage, size_t temp_col_num) {
  const size_t temp_lda = temp_col_num;
  ShrinkColsGpu<T, reduce_core_func>(row_num, temp_lda, x, col_num, col_num, temp_storage,
                                        temp_col_num, temp_lda);
  __syncthreads();
  while (temp_col_num > (1 << shift_size)) {
    size_t new_temp_col_num = temp_col_num >> shift_size;
    ShrinkColsGpu<T, reduce_core_func>(row_num, temp_lda, temp_storage, temp_col_num, temp_lda,
                                          temp_storage, new_temp_col_num, temp_lda);
    temp_col_num = new_temp_col_num;
    __syncthreads();
  }
  ShrinkColsGpu<T, reduce_core_func>(row_num, temp_lda, temp_storage, temp_col_num, temp_lda, y,
                                        1, 1);
}

template<typename T, T (*reduce_core_func)(const T, const T), size_t shift_size = 2>
void RowReduceGpuImpl(DeviceCtx* ctx, const size_t row_num, const size_t col_num, const T* x, T* y,
                     void* temp_storage, const size_t temp_storage_bytes) {
  CHECK_NOTNULL(temp_storage);
  CHECK_GT(temp_storage_bytes / sizeof(T), row_num);
  const size_t temp_col_num_shift =
      std::floor(std::log2(std::min(temp_storage_bytes / sizeof(T) / row_num, col_num)));
  const size_t temp_col_num = std::min(static_cast<size_t>(kCudaThreadsNumPerBlock),
                                       static_cast<size_t>(1 << temp_col_num_shift));
  RowReduceGpu<T, reduce_core_func>
      <<<BlocksNum4ThreadsNum(row_num * temp_col_num), kCudaThreadsNumPerBlock, 0,
         ctx->cuda_stream()>>>(row_num, col_num, x, y, static_cast<T*>(temp_storage), temp_col_num);
}

} // namespace

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
    const int m, const int n, const int k, const float alpha, const float* a, const float* b,
const float beta, float* c) {
  Gemm<float>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
    const int m, const int n, const int k, const double alpha, const double* a, const double* b,
    const double beta, double* c) {
  Gemm<double>(ctx, CblasRowMajor, trans_a, trans_b, m, n, k, alpha, a, b, beta, c);
}

GPU_KU_METHOD OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
    const int m, const int n, const int k, const float16 alpha, const float16* a, const float16* b,
    const float16 beta, float16* c) {
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
  ReluForwardGpuHalf
  <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
    n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
}
  
GPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const float* x, const float* y, const float* dy,
                           float* dx) {
  ReluBackwardGpu<float>
  <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}
  
GPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const double* x, const double* y, const double* dy,
                           double* dx) {
  ReluBackwardGpu<double>
  <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
}

GPU_KU_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const float16* x, const float16* y, const float16* dy,
                           float16* dx) {
ReluBackwardGpuHalf
<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
  n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy), reinterpret_cast<half*>(dx));
}

GPU_KU_METHOD RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const float* x, float* y,
  void* temp_storage, const size_t temp_storage_bytes) {
  RowReduceGpuImpl<float, NewReduceCoreMax>(ctx, row_num, col_num, x, y, temp_storage, temp_storage_bytes);                        
}

GPU_KU_METHOD RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const double* x, double* y,
  void* temp_storage, const size_t temp_storage_bytes) {
  RowReduceGpuImpl<double, NewReduceCoreMax>(ctx, row_num, col_num, x, y, temp_storage, temp_storage_bytes);                      
}

GPU_KU_METHOD RowMax(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const float16* x, float16* y,
  void* temp_storage, const size_t temp_storage_bytes) {
  RowReduceGpuImpl<half, NewReduceCoreMax>(ctx, row_num, col_num, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y), temp_storage, temp_storage_bytes);                      
}

GPU_KU_METHOD RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const float* x, float* y,
  void* temp_storage, const size_t temp_storage_bytes) {
  RowReduceGpuImpl<float, NewReduceCoreAdd>(ctx, row_num, col_num, x, y, temp_storage, temp_storage_bytes);
}

GPU_KU_METHOD RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const double* x, double* y,
  void* temp_storage, const size_t temp_storage_bytes) {
  RowReduceGpuImpl<double, NewReduceCoreAdd>(ctx, row_num, col_num, x, y, temp_storage, temp_storage_bytes);
}

GPU_KU_METHOD RowSum(DeviceCtx* ctx, const int64_t row_num, const int64_t col_num, const float16* x, float16* y,
  void* temp_storage, const size_t temp_storage_bytes) {
  RowReduceGpuImpl<half, NewReduceCoreMax>(ctx, row_num, col_num, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y), temp_storage, temp_storage_bytes);
}

} // namespace oneflow
