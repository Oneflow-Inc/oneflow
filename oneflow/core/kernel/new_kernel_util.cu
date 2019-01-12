#include <cub/cub.cuh>
#include <math.h>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void RsqrtGpu(const int64_t n, T* x, const float epsilon) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = 1.0 / std::sqrt(x[i] + epsilon); }
}

template<typename T>
__global__ void ExpGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::exp(x[i]); }
}

template<typename T>
__global__ void DivByConstParaPtrGpu(const int64_t n, T* x, const T* alpha_ptr) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = x[i] / (*alpha_ptr); }
}

template<typename T>
__global__ void DivGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] / y[i]; }
}

template<typename T>
__global__ void DivByConstParaGpu(const int64_t n, T* x, const T alpha) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = x[i] / alpha; }
}

template<typename T>
__global__ void ReplicateGpu(const int64_t n, T* y, const T* x) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = *x; }
}

template<typename T>
__global__ void MulGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y[i]; }
}

template<typename T>
__global__ void MulByScalarGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y[0]; }
}

template<typename T>
__global__ void AddByScalarGpu(const int64_t n, const T* x, const T y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] + y; }
}

template<typename T>
__global__ void MulByScalarParaGpu(const int64_t n, const T* x, const T y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y; }
}

template<typename T>
__global__ void ReciprocalGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = static_cast<T>(1.0) / x[i]; }
}

template<typename T>
__global__ void SquareGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = x[i] * x[i]; }
}

template<typename T>
__global__ void SqrtGpu(const int64_t n, const T* x, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = std::sqrt(x[i]); }
}

template<typename T>
__global__ void AxpyGpu(const int n, const T alpha, const T* x, const int incx, T* y,
                        const int incy) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i * incy] += alpha * x[i * incx]; }
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

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i]; }
}
template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i]; }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i];
  }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6, const T* in_7) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] = in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i];
  }
}

template<typename T>
__global__ void gpu_add(const int64_t n, T* out, const T* in_0, const T* in_1, const T* in_2,
                        const T* in_3, const T* in_4, const T* in_5, const T* in_6, const T* in_7,
                        const T* in_8) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    out[i] =
        in_0[i] + in_1[i] + in_2[i] + in_3[i] + in_4[i] + in_5[i] + in_6[i] + in_7[i] + in_8[i];
  }
}

template<typename T>
__global__ void gpu_set(const T value, T* addr) {
  *addr = value;
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

}  // namespace

#define NEW_KU_IF_GPU_METHOD(type_category) \
  template<typename T>                      \
  void NewKernelUtilIf<DeviceType::kGPU, T, \
                       typename std::enable_if<type_category<T>::value>::type>::

// GPU && Floating
NEW_KU_IF_GPU_METHOD(IsFloating)
InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                   Blob* blob, const std::string& data_format) {
  GpuInitializeWithConf<T>(ctx, initializer_conf, random_seed, blob, data_format);
}
NEW_KU_IF_GPU_METHOD(IsFloating)
InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                   Blob* blob) {
  InitializeWithConf(ctx, initializer_conf, random_seed, blob, "");
}
NEW_KU_IF_GPU_METHOD(IsFloating)
InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_dir,
                  Blob* blob, const std::string& bn_in_op, int32_t dim_num,
                  int64_t num_in_each_dim) {
  GpuInitializeWithDir<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                          num_in_each_dim);
}

// GPU && Integral
NEW_KU_IF_GPU_METHOD(IsIntegral)
InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                   Blob* blob, const std::string& data_format) {
  GpuInitializeWithConf<T>(ctx, initializer_conf, random_seed, blob, data_format);
}
NEW_KU_IF_GPU_METHOD(IsIntegral)
InitializeWithConf(DeviceCtx* ctx, const InitializerConf& initializer_conf, uint32_t random_seed,
                   Blob* blob) {
  InitializeWithConf(ctx, initializer_conf, random_seed, blob, "");
}
NEW_KU_IF_GPU_METHOD(IsIntegral)
InitializeWithDir(DeviceCtx* ctx, int32_t part_id, int32_t part_num, const std::string& model_dir,
                  Blob* blob, const std::string& bn_in_op, int32_t dim_num,
                  int64_t num_in_each_dim) {
  GpuInitializeWithDir<T>(ctx, part_id, part_num, model_dir, blob, bn_in_op, dim_num,
                          num_in_each_dim);
}

#undef NEW_KU_IF_GPU_METHOD

#define GPU_NEW_KU_IF_METHOD(type_category) \
  template<typename T>                      \
  void GpuNewKernelUtilIf<T, typename std::enable_if<type_category<T>::value>::type>::

GPU_NEW_KU_IF_METHOD(IsFloating)
Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape, const Shape& y_shape,
          const PbRf<int32_t>& permutation, const int64_t elem_cnt, const T* x, T* y) {
  CHECK_LE(y_shape.elem_cnt(), MaxVal<int32_t>::value);
  CHECK_EQ(num_axis, y_shape.NumAxes());
  CHECK_EQ(num_axis, x_shape.NumAxes());
  TransposeUtil<T>::SwitchTranspose(SwitchCase(num_axis), ctx, x_shape, y_shape, permutation,
                                    elem_cnt, x, y);
}

GPU_NEW_KU_IF_METHOD(IsIntegral)
Transpose(DeviceCtx* ctx, const int32_t num_axis, const Shape& x_shape, const Shape& y_shape,
          const PbRf<int32_t>& permutation, const int64_t elem_cnt, const T* x, T* y) {
  CHECK_LE(y_shape.elem_cnt(), MaxVal<int32_t>::value);
  CHECK_EQ(num_axis, y_shape.NumAxes());
  CHECK_EQ(num_axis, x_shape.NumAxes());
  TransposeUtil<T>::SwitchTranspose(SwitchCase(num_axis), ctx, x_shape, y_shape, permutation,
                                    elem_cnt, x, y);
}
#undef GPU_NEW_KU_IF_METHOD

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto) \
  template struct NewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

#define INSTANTIATE_FLOATING_KERNEL_UTIL(type_cpp, type_proto) \
  template struct FloatingNewKernelUtilIf<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_FLOATING_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
