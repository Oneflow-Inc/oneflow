#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

#define HALF_CHECK_LOG                      \
  printf("use half need nvcc arch >= 530"); \
  assert(false);

/*
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
#else
  HALF_CHECK_LOG
#endif // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
*/

namespace {

__inline__ __device__ half hone() { return __float2half(1.0); }

__inline__ __device__ half hzero() { return __float2half(0.0); }

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

template<typename T>
__global__ void gpu_set(const T value, T* addr) {
  *addr = value;
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

template<typename T>
__global__ void ReluForwardGpu(const int n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] > 0 ? x[i] : 0; }
}

template<typename T>
__global__ void ReluBackwardGpu(const int n, const T* y, const T* dy, T* dx) {
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = y[i] > 0 ? dy[i] : 0; }
}

__global__ void SigmoidForwardGpu(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = __hdiv(hone(), __hadd(hone(), hexp(__hneg(x[i])))); }
#else
  HALF_CHECK_LOG
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

__global__ void SigmoidBackwardGpu(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = __hmul(dy[i], __hmul(y[i], __hsub(hone(), y[i]))); }
#else
  HALF_CHECK_LOG
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

__global__ void TanHForwardGpu(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    half ex = hexp(x[i]);
    half e_x = hexp(__hneg(x[i]));
    y[i] = __hdiv(__hsub(ex, e_x), __hadd(ex, e_x));
  }
#else
  HALF_CHECK_LOG
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void TanHBackwardGpu(const int n, const half* y, const half* dy, half* dx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { dx[i] = __hmul(dy[i], __hsub(hone(), __hmul(y[i], y[i]))); }
#else
  HALF_CHECK_LOG
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void ReluForwardGpu(const int n, const half* x, half* y) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) {
    if (__hgt(x[i], hzero())) {
      y[i] = x[i];
    } else {
      y[i] = hzero();
    }
  }
#else
  HALF_CHECK_LOG
#endif /* __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__) */
}

__global__ void ReluBackwardGpu(const int n, const half* y, const half* dy, half* dx) {
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
  HALF_CHECK_LOG
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void AxpyHalfGpu(const int n, const half alpha, const half* x, const int incx, half* y,
                            const int incy) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { y[i * incy] = __hfma(alpha, x[i * incx], y[i * incy]); }
#else
  HALF_CHECK_LOG
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void ScalHalfGpu(const int n, const half alpha, half* x, const int incx) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { x[i * incx] = __hmul(alpha, x[i * incx]); }
#else
  HALF_CHECK_LOG
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

__global__ void Float2HalfGpu(const int n, const float* src, half* dst) {
  CUDA_1D_KERNEL_LOOP(i, n) { dst[i] = __float2half(src[i]); }
}

__global__ void Half2FloatGpu(const int n, const half* src, float* dst) {
  CUDA_1D_KERNEL_LOOP(i, n) { dst[i] = __half2float(src[i]); }
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
void InitializeWithConfGpu(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                           uint32_t random_seed, Blob* blob, const std::string& data_format) {
  BEFORE_CPU_INITIALIZE();
  // synchronous initialize the host blob
  NewKernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, initializer_conf, random_seed,
                                                         host_blob.get(), data_format);
  AFTER_CPU_INITIALIZE();
}

template<typename T>
void InitializeWithDirGpu(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                          const std::string& model_dir, Blob* blob, const std::string& bn_in_op,
                          int32_t dim_num, int64_t num_in_each_dim) {
  BEFORE_CPU_INITIALIZE();
  NewKernelUtil<DeviceType::kCPU, T>::InitializeWithDir(
      ctx, part_id, part_num, model_dir, host_blob.get(), bn_in_op, dim_num, num_in_each_dim);
  AFTER_CPU_INITIALIZE();
}

}  // namespace

// GPU && Floating
template<typename T>
struct NewKernelUtilIf<DeviceType::kGPU, T, typename std::enable_if<IsFloating<T>::value>::type> {
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c) {
    const int lda = (trans_a == CblasNoTrans) ? k : m;
    const int ldb = (trans_b == CblasNoTrans) ? n : k;
    const int ldc = n;

    FloatingNewKernelUtilIf<DeviceType::kGPU, T>::Gemm(ctx, CblasRowMajor, trans_a, trans_b, m, n,
                                                       k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format) {
    InitializeWithConfGpu<T>(ctx, initializer_conf, random_seed, blob, data_format);
  }
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim) {
    InitializeWithDirGpu<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                            num_in_each_dim);
  }
  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    SigmoidForwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
  }
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx) {
    SigmoidBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
  }
  static void TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    TanHForwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
  }
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    TanHBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
  }
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    ReluForwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y);
  }
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    ReluBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, y, dy, dx);
  }
  static void Set(DeviceCtx* ctx, const T value, T* addr) {
    gpu_set<T><<<1, 1, 0, ctx->cuda_stream()>>>(value, addr);
  }
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy) {
    cublas_axpy<T>(ctx->cublas_pmh_handle(), n, &alpha, x, incx, y, incy);
  }
  static void Scal(DeviceCtx* ctx, const int n, const T alpha, T* x, const int incx) {
    cublas_scal<T>(ctx->cublas_pmh_handle(), n, &alpha, x, incx);
  }
};

// GPU && Integral
template<typename T>
struct NewKernelUtilIf<DeviceType::kGPU, T, typename std::enable_if<IsIntegral<T>::value>::type> {
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format) {
    InitializeWithConfGpu<T>(ctx, initializer_conf, random_seed, blob, data_format);
  }
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim) {
    InitializeWithDirGpu<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                            num_in_each_dim);
  }
  static void Set(DeviceCtx* ctx, const T value, T* addr) {
    gpu_set<T><<<1, 1, 0, ctx->cuda_stream()>>>(value, addr);
  }
};

// GPU && Float16
template<typename T>
struct NewKernelUtilIf<DeviceType::kGPU, T, typename std::enable_if<IsFloat16<T>::value>::type> {
  static void OFGemm(DeviceCtx* ctx, enum CBLAS_TRANSPOSE trans_a, enum CBLAS_TRANSPOSE trans_b,
                     const int m, const int n, const int k, const T alpha, const T* a, const T* b,
                     const T beta, T* c) {
    const int lda = (trans_a == CblasNoTrans) ? k : m;
    const int ldb = (trans_b == CblasNoTrans) ? n : k;
    const int ldc = n;

    Float16NewKernelUtilIf<DeviceType::kGPU, T>::HGemm(ctx, CblasRowMajor, trans_a, trans_b, m, n,
                                                       k, alpha, a, lda, b, ldb, beta, c, ldc);
  }
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format) {
    InitializeWithConfGpu<T>(ctx, initializer_conf, random_seed, blob, data_format);
  }
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim) {
    InitializeWithDirGpu<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                            num_in_each_dim);
  }
  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    SigmoidForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  }
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx) {
    SigmoidBackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx));
  }
  static void TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    TanHForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  }
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    TanHBackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx));
  }
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
    ReluForwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(x), reinterpret_cast<half*>(y));
  }
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    ReluBackwardGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(y), reinterpret_cast<const half*>(dy),
        reinterpret_cast<half*>(dx));
  }
  static void Set(DeviceCtx* ctx, const T value, T* addr) {
    gpu_set<half>
        <<<1, 1, 0, ctx->cuda_stream()>>>(static_cast<half>(value), reinterpret_cast<half*>(addr));
  }
  static void Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx, T* y,
                   const int incy) {
    // half ha;
    // ha.setx(alpha.getx());
    AxpyHalfGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, float16_2half(alpha), reinterpret_cast<const half*>(x), incx, reinterpret_cast<half*>(y),
        incy);
  }
  static void Scal(DeviceCtx* ctx, const int n, const T alpha, T* x, const int incx) {
    // half ha;
    // ha.setx(alpha.getx());
    ScalHalfGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, float16_2half(alpha), reinterpret_cast<half*>(x), incx);
  }
};

template<typename T>
struct FloatingNewKernelUtilIf<DeviceType::kGPU, T> {
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c,
                   const int ldc) {  // TODO: wrong CUBLAS_OP_N
    cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
    cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
    cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha, b,
                   ldb, a, lda, &beta, c, ldc);
  }
};

template<typename T>
struct Float16NewKernelUtilIf<DeviceType::kGPU, T> {
  static void HGemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                    const enum CBLAS_TRANSPOSE trans_a, const enum CBLAS_TRANSPOSE trans_b,
                    const int m, const int n, const int k, const T alpha, const T* a, const int lda,
                    const T* b, const int ldb, const T beta, T* c, const int ldc) {
    cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
    cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
    CudaCheck(cublasHgemm(ctx->cublas_tensor_op_math_handle(), cublas_trans_b, cublas_trans_a, n, m,
                          k, reinterpret_cast<const half*>(&alpha),
                          reinterpret_cast<const half*>(b), ldb, reinterpret_cast<const half*>(a),
                          lda, reinterpret_cast<const half*>(&beta), reinterpret_cast<half*>(c),
                          ldc));
  }
  static void Half2Float(DeviceCtx* ctx, const int n, const T* src, float* dst) {
    Half2FloatGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, reinterpret_cast<const half*>(src), dst);
  }
  static void Float2Half(DeviceCtx* ctx, const int n, const float* src, T* dst) {
    Float2HalfGpu<<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
        n, src, reinterpret_cast<half*>(dst));
  }
};

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

#define INSTANTIATE_FLOATING_KERNEL_UTIL(type_cpp, type_proto) \
  template struct FloatingNewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_FLOATING_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

#define INSTANTIATE_FLOAT16_KERNEL_UTIL(type_cpp, type_proto) \
  template struct Float16NewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_FLOAT16_KERNEL_UTIL, FLOAT16_DATA_TYPE_SEQ);

}  // namespace oneflow
