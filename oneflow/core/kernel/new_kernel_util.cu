#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
void GpuInitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                           uint32_t random_seed, Blob* blob, const std::string& data_format) {
  BEFORE_CPU_INITIALIZE();
  // synchronous initialize the host blob
  NewKernelUtil<DeviceType::kCPU, T>::InitializeWithConf(nullptr, initializer_conf, random_seed,
                                                         host_blob.get(), data_format);
  AFTER_CPU_INITIALIZE();
}

template<typename T>
void GpuInitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
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

template<typename T>
__global__ void CopyColsRegionGpu(const int64_t row_num, const int64_t col_num, const T* x,
                                  const int64_t x_col_offset, const int64_t x_lda, T* y,
                                  const int64_t y_col_offset, const int64_t y_lda) {
  CUDA_1D_KERNEL_LOOP(index, row_num * col_num) {
    const int64_t i = index / col_num;
    const int64_t j = index % col_num;
    y[i * y_lda + y_col_offset + j] = x[i * x_lda + x_col_offset + j];
  }
}

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
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format) {
    GpuInitializeWithConf<T>(ctx, initializer_conf, random_seed, blob, data_format);
  }
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob) {
    InitializeWithConf(ctx, initializer_conf, random_seed, blob, "");
  }
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim) {
    GpuInitializeWithDir<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
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
  }
};

// GPU && Integral
template<typename T>
struct NewKernelUtilIf<DeviceType::kGPU, T, typename std::enable_if<IsIntegral<T>::value>::type> {
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob, const std::string& data_format) {
    GpuInitializeWithConf<T>(ctx, initializer_conf, random_seed, blob, data_format);
  }
  static void InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf,
                                 uint32_t random_seed, Blob* blob) {
    InitializeWithConf(ctx, initializer_conf, random_seed, blob, "");
  }
  static void InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num,
                                const std::string& model_dir, Blob* blob,
                                const std::string& bn_in_op, int32_t dim_num,
                                int64_t num_in_each_dim) {
    GpuInitializeWithDir<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
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
  }
};

template<typename T>
struct FloatingNewKernelUtilIf<DeviceType::kGPU, T> {
  static void Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order, const enum CBLAS_TRANSPOSE trans_a,
                   const enum CBLAS_TRANSPOSE trans_b, const int m, const int n, const int k,
                   const T alpha, const T* a, const int lda, const T* b, const int ldb,
                   const T beta, T* c,
                   const int ldc) {  // TODO: wrong CUBLAS_OP_N
    cublasOperation_t cublas_trans_a = cublasOperation_t::CUBLAS_OP_N;
    cublasOperation_t cublas_trans_b = cublasOperation_t::CUBLAS_OP_N;
    cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m, k, &alpha, b,
                   ldb, a, lda, &beta, c, ldc);
  }
};

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

#define INSTANTIATE_FLOATING_KERNEL_UTIL(type_cpp, type_proto) \
  template struct FloatingNewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_FLOATING_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
