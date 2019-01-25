#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

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
void Transpose(DeviceCtx* ctx, const Shape& x_shape, const Shape& y_shape,
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
  DEFINE_STATIC_SWITCH_FUNC(void, Transpose, MAKE_TRANSPOSE_SWITCH_ENTRY,
                            MAKE_NDIM_CTRV_SEQ(DIM_SEQ));
};
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
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y) {
    CHECK_LE(y_shape.elem_cnt(), MaxVal<int32_t>::value);
    CHECK_EQ(num_axis, y_shape.NumAxes());
    CHECK_EQ(num_axis, x_shape.NumAxes());
    TransposeUtil<T>::SwitchTranspose(SwitchCase(num_axis), ctx, x_shape, y_shape, permutation,
                                      elem_cnt, x, y);
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
  static void Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape,
                        const Shape& y_shape, const PbRf<int32_t>& permutation,
                        const int64_t elem_cnt, const T* x, T* y) {
    CHECK_LE(y_shape.elem_cnt(), MaxVal<int32_t>::value);
    CHECK_EQ(num_axis, y_shape.NumAxes());
    CHECK_EQ(num_axis, x_shape.NumAxes());
    TransposeUtil<T>::SwitchTranspose(SwitchCase(num_axis), ctx, x_shape, y_shape, permutation,
                                      elem_cnt, x, y);
  static void Sigmoid(DeviceCtx* ctx, const int64_t n, const T* x, T* y) { UNIMPLEMENTED(); }
  static void SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                              T* dx) {
    UNIMPLEMENTED();
  }
  static void TanH(DeviceCtx* ctx, const int64_t n, const T* x, T* y) { UNIMPLEMENTED(); }
  static void TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    UNIMPLEMENTED();
  }
  static void Relu(DeviceCtx* ctx, const int64_t n, const T* x, T* y) { UNIMPLEMENTED(); }
  static void ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y, const T* dy,
                           T* dx) {
    UNIMPLEMENTED();
  }
  static void Set(DeviceCtx* ctx, const T value, T* addr) {
    gpu_set<half>
        <<<1, 1, 0, ctx->cuda_stream()>>>(static_cast<half>(value), reinterpret_cast<half*>(addr));
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
    CudaCheck(cublasHgemm(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k,
                          reinterpret_cast<const half*>(&alpha), reinterpret_cast<const half*>(b),
                          ldb, reinterpret_cast<const half*>(a), lda,
                          reinterpret_cast<const half*>(&beta), reinterpret_cast<half*>(c), ldc));
  }
};

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ FLOAT16_DATA_TYPE_SEQ);

#define INSTANTIATE_FLOATING_KERNEL_UTIL(type_cpp, type_proto) \
  template struct FloatingNewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_FLOATING_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
