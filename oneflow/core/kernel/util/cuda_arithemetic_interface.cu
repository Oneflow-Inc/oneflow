#include "oneflow/core/kernel/util/cuda_arithemetic_interface.h"
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/kernel/util/host_arithemetic_interface.h"
#include "oneflow/core/kernel/new_kernel_util.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"

namespace oneflow {

namespace {

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
  CHECK_LE(y_shape.elem_cnt(), GetMaxVal<int32_t>());
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

#define TRANSPOSE_CHECK                               \
  CHECK_LE(y_shape.elem_cnt(), GetMaxVal<int32_t>()); \
  CHECK_EQ(num_axis, y_shape.NumAxes());              \
  CHECK_EQ(num_axis, x_shape.NumAxes())

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const Shape& x_shape, const Shape& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const float* x, float* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<float>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                            permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const Shape& x_shape, const Shape& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const double* x,
                                                double* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<double>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape,
                                             permutation, elem_cnt, x, y);
}

void ArithemeticIf<DeviceType::kGPU>::Transpose(DeviceCtx* ctx, const int32_t num_axis,
                                                const Shape& x_shape, const Shape& y_shape,
                                                const PbRf<int32_t>& permutation,
                                                const int64_t elem_cnt, const float16* x,
                                                float16* y) {
  TRANSPOSE_CHECK;
  TransposeUtil<half>::SwitchTransposeImpl(SwitchCase(num_axis), ctx, x_shape, y_shape, permutation,
                                           elem_cnt, reinterpret_cast<const half*>(x),
                                           reinterpret_cast<half*>(y));
}

#undef TRANSPOSE_CHECK

// create temporary host blob store initializer result
#define BEFORE_CPU_INITIALIZE()                                     \
  RtBlobDesc blob_desc(blob->blob_desc().blob_desc_proto());        \
  char* host_raw_dptr = nullptr;                                    \
  CudaCheck(cudaMallocHost(&host_raw_dptr, blob->TotalByteSize())); \
  std::unique_ptr<Blob> host_blob;                                  \
  host_blob.reset(new Blob(nullptr, &blob_desc, host_raw_dptr));

// asynchronous copy to device
#define AFTER_CPU_INITIALIZE()                                                          \
  Memcpy<DeviceType::kGPU>(ctx, blob->mut_dptr(), host_blob->dptr(),                    \
                           blob->ByteSizeOfDataContentField(), cudaMemcpyHostToDevice); \
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));                                 \
  CudaCheck(cudaFreeHost(host_raw_dptr));

void ArithemeticIf<DeviceType::kGPU>::InitializeWithConstConf(
    DeviceCtx* ctx, const ConstantInitializerConf& initializer_conf, Blob* blob) {
  BEFORE_CPU_INITIALIZE();
  // synchronous initialize the host blob
  ArithemeticIf<DeviceType::kCPU>::InitializeWithConstConf(nullptr, initializer_conf,
                                                           host_blob.get());
  AFTER_CPU_INITIALIZE();
}

#undef BEFORE_CPU_INITIALIZE
#undef AFTER_CPU_INITIALIZE

namespace {

template<typename T>
__global__ void MulByScalarGpu(const int64_t n, const T* x, const T y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y; }
}

template<>
__global__ void MulByScalarGpu<half>(const int64_t n, const half* x, const half y, half* z) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = __hmul(x[i], y); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

}  // namespace

#define MUL_BY_SCALAR(T)                                                                           \
  void ArithemeticIf<DeviceType::kGPU>::MulByScalar(DeviceCtx* ctx, const int64_t n, const T* x,   \
                                                    const T y, T* z) {                             \
    MulByScalarGpu<T>                                                                              \
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z); \
  }

MUL_BY_SCALAR(float)
MUL_BY_SCALAR(double)
MUL_BY_SCALAR(int32_t)
MUL_BY_SCALAR(int64_t)

#undef MUL_BY_SCALAR

void ArithemeticIf<DeviceType::kGPU>::MulByScalar(DeviceCtx* ctx, const int64_t n, const float16* x,
                                                  const float16 y, float16* z) {
  MulByScalarGpu<half><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), float16_2half(y), reinterpret_cast<half*>(z));
}

namespace {

template<typename T>
__global__ void AddByScalarGpu(const int64_t n, const T* x, const T y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] + y; }
}

template<>
__global__ void AddByScalarGpu<half>(const int64_t n, const half* x, const half y, half* z) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = __hadd(x[i], y); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

}  // namespace

#define ADD_BY_SCALAR(T)                                                                           \
  void ArithemeticIf<DeviceType::kGPU>::AddByScalar(DeviceCtx* ctx, const int64_t n, const T* x,   \
                                                    const T y, T* z) {                             \
    AddByScalarGpu<T>                                                                              \
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z); \
  }

ADD_BY_SCALAR(float)
ADD_BY_SCALAR(double)
ADD_BY_SCALAR(int32_t)
ADD_BY_SCALAR(int64_t)

#undef ADD_BY_SCALAR

void ArithemeticIf<DeviceType::kGPU>::AddByScalar(DeviceCtx* ctx, const int64_t n, const float16* x,
                                                  const float16 y, float16* z) {
  AddByScalarGpu<half><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), float16_2half(y), reinterpret_cast<half*>(z));
}

namespace {

template<typename T>
__global__ void MulGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y[i]; }
}

template<>
__global__ void MulGpu<half>(const int64_t n, const half* x, const half* y, half* z) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = __hmul(x[i], y[i]); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

}  // namespace

#define MUL(T)                                                                                     \
  void ArithemeticIf<DeviceType::kGPU>::Mul(DeviceCtx* ctx, const int64_t n, const T* x,           \
                                            const T* y, T* z) {                                    \
    MulGpu<T>                                                                                      \
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z); \
  }

MUL(float)
MUL(double)

#undef MUL

void ArithemeticIf<DeviceType::kGPU>::Mul(DeviceCtx* ctx, const int64_t n, const float16* x,
                                          const float16* y, float16* z) {
  MulGpu<half><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
      n, reinterpret_cast<const half*>(x), reinterpret_cast<const half*>(y),
      reinterpret_cast<half*>(z));
}

namespace {

template<typename T>
__global__ void MulByGpuScalarGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y[0]; }
}

template<>
__global__ void MulByGpuScalarGpu<half>(const int64_t n, const half* x, const half* y, half* z) {
#if __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = __hmul(x[i], y[0]); }
#else
  HALF_CHECK_FAILED;
#endif  // __CUDA_ARCH__ >= 530 || !defined(__CUDA_ARCH__)
}

}  // namespace

#define MUL_BY_GPU_SCALAR(T)                                                                       \
  void ArithemeticIf<DeviceType::kGPU>::MulByGpuScalar(DeviceCtx* ctx, const int64_t n,            \
                                                       const T* x, const T* y, T* z) {             \
    MulByGpuScalarGpu<T>                                                                           \
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z); \
  }

MUL_BY_GPU_SCALAR(float)
MUL_BY_GPU_SCALAR(double)
MUL_BY_GPU_SCALAR(int32_t)
MUL_BY_GPU_SCALAR(int64_t)

#undef MUL_BY_GPU_SCALAR

void ArithemeticIf<DeviceType::kGPU>::MulByGpuScalar(DeviceCtx* ctx, const int64_t n,
                                                     const float16* x, const float16* y,
                                                     float16* z) {
  MulByGpuScalarGpu<half>
      <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(
          n, reinterpret_cast<const half*>(x), reinterpret_cast<const half*>(y),
          reinterpret_cast<half*>(z));
}

namespace {

template<typename T>
__global__ void PowByIntScalarGpu(const int64_t n, const T* x, const int32_t y, T* z);

template<>
__global__ void PowByIntScalarGpu<float>(const int64_t n, const float* x, const int32_t y,
                                         float* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = powf(x[i], y); }
}

template<>
__global__ void PowByIntScalarGpu<double>(const int64_t n, const double* x, const int32_t y,
                                          double* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = pow(x[i], y); }
}

template<>
__global__ void PowByIntScalarGpu<int32_t>(const int64_t n, const int32_t* x, const int32_t y,
                                           int32_t* z) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    z[i] = static_cast<int32_t>(powf(static_cast<float>(x[i]), static_cast<float>(y)));
  }
}

template<>
__global__ void PowByIntScalarGpu<int64_t>(const int64_t n, const int64_t* x, const int32_t y,
                                           int64_t* z) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    z[i] = static_cast<int64_t>(pow(static_cast<double>(x[i]), static_cast<double>(y)));
  }
}

}  // namespace

#define POW_BY_INT_SCALAR(T)                                                                       \
  void ArithemeticIf<DeviceType::kGPU>::PowByIntScalar(DeviceCtx* ctx, const int64_t n,            \
                                                       const T* x, const int32_t y, T* z) {        \
    PowByIntScalarGpu<T>                                                                           \
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0, ctx->cuda_stream()>>>(n, x, y, z); \
  }

POW_BY_INT_SCALAR(float)
POW_BY_INT_SCALAR(double)
POW_BY_INT_SCALAR(int32_t)
POW_BY_INT_SCALAR(int64_t)

#undef POW_BY_INT_SCALAR

}  // namespace oneflow
