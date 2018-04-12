#include <cub/cub.cuh>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/kernel/kernel_util.cuh"

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
__global__ void DivGpu(const int64_t n, T* x, const T* alpha_ptr) {
  CUDA_1D_KERNEL_LOOP(i, n) { x[i] = x[i] / (*alpha_ptr); }
}

template<typename T>
__global__ void MulGpu(const int64_t n, const T* x, const T* y, T* z) {
  CUDA_1D_KERNEL_LOOP(i, n) { z[i] = x[i] * y[i]; }
}

template<typename T>
__global__ void AxpyGpu(const int n, const T alpha, const T* x, const int incx,
                        T* y, const int incy) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i * incy] += alpha * x[i * incx]; }
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

const int32_t kMaxDim = OF_PP_SEQ_SIZE(DIM_SEQ);

struct Int32Array {
  int32_t val[kMaxDim];
};

struct Int64Array {
  int64_t val[kMaxDim];
};

__device__ void ComputeOffset(const int32_t num_axis, const int64_t* x_dims,
                              const int32_t* permutation, int64_t* x_strides) {
  int64_t buff[kMaxDim];
  int64_t cur_stride = 1;
  for (int32_t i = num_axis - 1; i >= 0; --i) {
    buff[i] = cur_stride;
#if __CUDA_ARCH__ >= 350
    cur_stride *= __ldg(x_dims + i);
#else
    cur_stride *= x_dims[i];
#endif
  }
  for (int32_t i = 0; i < num_axis; ++i) {
#if __CUDA_ARCH__ >= 350
    x_strides[i] = buff[__ldg(permutation + i)];
#else
    x_strides[i] = buff[permutation[i]];
#endif
  }
}

__device__ int64_t GetXIndex(const int32_t num_axis, const int64_t* y_shape,
                             const int64_t* x_strides, int64_t y_idx) {
  int64_t x_idx = 0;
  for (int32_t i = num_axis - 1; i >= 0 && y_idx > 0; --i) {
    x_idx += (y_idx % y_shape[i]) * x_strides[i];
    y_idx /= y_shape[i];
  }
  return x_idx;
}

template<typename T>
__global__ void TransposeGpu(const int32_t num_axis, const Int64Array x_shape,
                             const Int64Array y_shape,
                             const Int32Array permutation,
                             const int64_t elem_cnt, const T* x, T* y) {
  __shared__ int64_t x_strides[kMaxDim];
  __shared__ int64_t x_dims_shared[kMaxDim];
  __shared__ int64_t y_dims_shared[kMaxDim];
  __shared__ int32_t perm_shared[kMaxDim];
  const int32_t tid = threadIdx.x;
  if (tid < num_axis) {
    x_dims_shared[tid] = x_shape.val[tid];
    y_dims_shared[tid] = y_shape.val[tid];
    perm_shared[tid] = permutation.val[tid];
  }
  __syncthreads();
  if (tid == 0) {
    ComputeOffset(num_axis, x_dims_shared, perm_shared, x_strides);
  }
  __syncthreads();
  CUDA_1D_KERNEL_LOOP(y_idx, elem_cnt) {
    const int64_t x_idx = GetXIndex(num_axis, y_dims_shared, x_strides, y_idx);
#if __CUDA_ARCH__ >= 350
    y[y_idx] = __ldg(x + x_idx);
#else
    y[y_idx] = x[x_idx];
#endif
  }
}

}  // namespace

template<>
void Memcpy<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const void* src,
                              size_t sz, cudaMemcpyKind kind) {
  CudaCheck(cudaMemcpyAsync(dst, src, sz, kind, ctx->cuda_stream()));
}

template<>
void Memset<DeviceType::kGPU>(DeviceCtx* ctx, void* dst, const char value,
                              size_t sz) {
  CudaCheck(cudaMemsetAsync(dst, value, sz, ctx->cuda_stream()));
}

#define MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY(func_name, T) \
  cub::DeviceReduce::func_name<T*, T*>
DEFINE_STATIC_SWITCH_FUNC(cudaError_t, Sum, MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY,
                          MAKE_DATA_TYPE_CTRV_SEQ(FLOATING_DATA_TYPE_SEQ));

size_t GetTmpSizeForReduceSum(DataType data_type, int64_t sum_elem_num) {
  size_t tmp_storage_size;
  SwitchSum(SwitchCase(data_type), nullptr, tmp_storage_size, nullptr, nullptr,
            sum_elem_num);
  return tmp_storage_size;
}

#undef MAKE_CUB_DEVICE_REDUCE_SWITCH_ENTRY

#define KU_IF_METHOD                     \
  template<typename T, typename Derived> \
  void GpuKernelUtilIf<T, Derived>::

KU_IF_METHOD Max(DeviceCtx* ctx, const int64_t n, const T* x, T* max_ptr,
                 T* temp_storage, size_t temp_storage_bytes) {
  CudaCheck(cub::DeviceReduce::Max(temp_storage, temp_storage_bytes, x, max_ptr,
                                   n, ctx->cuda_stream()));
}
KU_IF_METHOD Sum(DeviceCtx* ctx, const int64_t n, const T* x, T* sum_ptr,
                 T* temp_storage, size_t temp_storage_bytes) {
  CudaCheck(cub::DeviceReduce::Sum(temp_storage, temp_storage_bytes, x, sum_ptr,
                                   n, ctx->cuda_stream()));
}
KU_IF_METHOD Transpose(DeviceCtx* ctx, const int32_t num_axis,
                       const Shape& x_shape, const Shape& y_shape,
                       const PbRf<int32_t>& permutation, const int64_t elem_cnt,
                       const T* x, T* y) {
  CHECK_LE(num_axis, kMaxDim);
  Int64Array x_shape_struct;
  Int64Array y_shape_struct;
  Int32Array perm_struct;
  FOR_RANGE(int32_t, i, 0, num_axis) {
    x_shape_struct.val[i] = x_shape.At(i);
    y_shape_struct.val[i] = y_shape.At(i);
    perm_struct.val[i] = permutation[i];
  }
  TransposeGpu<T><<<BlocksNum4ThreadsNum(elem_cnt), kCudaThreadsNumPerBlock, 0,
                    ctx->cuda_stream()>>>(
      num_axis, x_shape_struct, y_shape_struct, perm_struct, elem_cnt, x, y);
}

#define KU_FLOATING_METHOD             \
  template<typename T>                 \
  void KernelUtil<DeviceType::kGPU, T, \
                  typename std::enable_if<IsFloating<T>::value>::type>::

KU_FLOATING_METHOD Dot(DeviceCtx* ctx, const int n, const T* x, const int incx,
                       const T* y, const int incy, T* result) {
  cublas_dot<T>(ctx->cublas_pmd_handle(), n, x, incx, y, incy, result);
}
KU_FLOATING_METHOD Copy(DeviceCtx* ctx, const int n, const T* x, const int incx,
                        T* y, const int incy) {
  cublas_copy<T>(ctx->cublas_pmh_handle(), n, x, incx, y, incy);
}
KU_FLOATING_METHOD Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x,
                        const int incx, T* y, const int incy) {
  cublas_axpy<T>(ctx->cublas_pmh_handle(), n, &alpha, x, incx, y, incy);
}
KU_FLOATING_METHOD Axpy(DeviceCtx* ctx, const int n, const T* alpha, const T* x,
                        const int incx, T* y, const int incy) {
  cublas_axpy<T>(ctx->cublas_pmd_handle(), n, alpha, x, incx, y, incy);
}
KU_FLOATING_METHOD Scal(DeviceCtx* ctx, const int n, const T alpha, T* x,
                        const int incx) {
  cublas_scal<T>(ctx->cublas_pmh_handle(), n, &alpha, x, incx);
}
KU_FLOATING_METHOD Scal(DeviceCtx* ctx, const int n, const T* alpha, T* x,
                        const int incx) {
  cublas_scal<T>(ctx->cublas_pmd_handle(), n, alpha, x, incx);
}
KU_FLOATING_METHOD Gemv(DeviceCtx* ctx, const enum CBLAS_TRANSPOSE trans, int m,
                        int n, const T alpha, const T* a, int lda, const T* x,
                        const int incx, const T beta, T* y, const int incy) {
  cublasOperation_t cublas_trans = CblasTrans2CublasTrans(trans);
  cublas_gemv<T>(ctx->cublas_pmh_handle(), cublas_trans, n, m, &alpha, a, lda,
                 x, incx, &beta, y, incy);
}
KU_FLOATING_METHOD Gemm(DeviceCtx* ctx, const enum CBLAS_ORDER order,
                        const enum CBLAS_TRANSPOSE trans_a,
                        const enum CBLAS_TRANSPOSE trans_b, const int m,
                        const int n, const int k, const T alpha, const T* a,
                        const int lda, const T* b, const int ldb, const T beta,
                        T* c, const int ldc) {
  cublasOperation_t cublas_trans_a = CblasTrans2CublasTrans(trans_a);
  cublasOperation_t cublas_trans_b = CblasTrans2CublasTrans(trans_b);
  cublas_gemm<T>(ctx->cublas_pmh_handle(), cublas_trans_b, cublas_trans_a, n, m,
                 k, &alpha, b, ldb, a, lda, &beta, c, ldc);
}

KU_FLOATING_METHOD Exp(DeviceCtx* ctx, const int64_t n, const T* x, T* y) {
  ExpGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
              ctx->cuda_stream()>>>(n, x, y);
}
KU_FLOATING_METHOD Div(DeviceCtx* ctx, const int64_t n, T* x, const T* alpha) {
  DivGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
              ctx->cuda_stream()>>>(n, x, alpha);
}
KU_FLOATING_METHOD Mul(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                       T* z) {
  MulGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
              ctx->cuda_stream()>>>(n, x, y, z);
}
KU_FLOATING_METHOD Rsqrt(DeviceCtx* ctx, const int64_t n, T* x,
                         const float epsilon) {
  RsqrtGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                ctx->cuda_stream()>>>(n, x, epsilon);
}

#define CREATE_FORWARD_TENSOR_AND_ACTIVATION_DESCRIPTOR(mode)               \
  CudnnTensorDesc x_desc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, n, 1, 1, \
                         1);                                                \
  CudnnTensorDesc y_desc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, n, 1, 1, \
                         1);                                                \
  CudnnActivationDesc act_desc(mode, CUDNN_PROPAGATE_NAN, 0.0);

#define FORWARD_COMPUTE_ACTIVATION(mode)                                \
  CREATE_FORWARD_TENSOR_AND_ACTIVATION_DESCRIPTOR(mode);                \
  CudaCheck(cudnnActivationForward(ctx->cudnn_handle(), act_desc.Get(), \
                                   OnePtr<T>::value, x_desc.Get(), x,   \
                                   ZeroPtr<T>::value, y_desc.Get(), y));

#define CREATE_BACKWARD_TENSOR_AND_ACTIVATION_DESCRIPTOR(mode)               \
  CREATE_FORWARD_TENSOR_AND_ACTIVATION_DESCRIPTOR(mode);                     \
  CudnnTensorDesc dx_desc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, n, 1, 1, \
                          1);                                                \
  CudnnTensorDesc dy_desc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, n, 1, 1, 1);

#define BACKWARD_COMPUTE_ACTIVATION(mode)                                \
  CREATE_BACKWARD_TENSOR_AND_ACTIVATION_DESCRIPTOR(mode);                \
  CudaCheck(cudnnActivationBackward(ctx->cudnn_handle(), act_desc.Get(), \
                                    OnePtr<T>::value, y_desc.Get(), y,   \
                                    dy_desc.Get(), dy, x_desc.Get(), x,  \
                                    ZeroPtr<T>::value, dx_desc.Get(), dx));

KU_FLOATING_METHOD Sigmoid(DeviceCtx* ctx, int64_t n, const T* x, T* y){
    FORWARD_COMPUTE_ACTIVATION(CUDNN_ACTIVATION_SIGMOID)} KU_FLOATING_METHOD
    SigmoidBackward(DeviceCtx* ctx, const int64_t n, const T* x, const T* y,
                    const T* dy, T* dx) {
  BACKWARD_COMPUTE_ACTIVATION(CUDNN_ACTIVATION_SIGMOID);
}
KU_FLOATING_METHOD TanH(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
  FORWARD_COMPUTE_ACTIVATION(CUDNN_ACTIVATION_TANH);
}
KU_FLOATING_METHOD TanHBackward(DeviceCtx* ctx, const int64_t n, const T* x,
                                const T* y, const T* dy, T* dx) {
  BACKWARD_COMPUTE_ACTIVATION(CUDNN_ACTIVATION_TANH);
}
KU_FLOATING_METHOD Relu(DeviceCtx* ctx, int64_t n, const T* x, T* y) {
  FORWARD_COMPUTE_ACTIVATION(CUDNN_ACTIVATION_RELU);
}
KU_FLOATING_METHOD ReluBackward(DeviceCtx* ctx, const int64_t n, const T* x,
                                const T* y, const T* dy, T* dx) {
  BACKWARD_COMPUTE_ACTIVATION(CUDNN_ACTIVATION_RELU);
}

// create temporary host blob store initializer result
#define BEFORE_CPU_INITIALIZE()                                     \
  BlobDesc blob_desc = BlobDesc(blob->blob_desc());                 \
  char* host_raw_dptr = nullptr;                                    \
  CudaCheck(cudaMallocHost(&host_raw_dptr, blob->TotalByteSize())); \
  std::unique_ptr<Blob> host_blob;                                  \
  host_blob.reset(                                                  \
      NewBlob(nullptr, &blob_desc, host_raw_dptr, nullptr, DeviceType::kGPU));

// asynchronous copy to device
#define AFTER_CPU_INITIALIZE()                                       \
  Memcpy<DeviceType::kGPU>(ctx, blob->mut_dptr(), host_blob->dptr(), \
                           blob->ByteSizeOfDataContentField(),       \
                           cudaMemcpyHostToDevice);                  \
  CudaCheck(cudaStreamSynchronize(ctx->cuda_stream()));              \
  CudaCheck(cudaFreeHost(host_raw_dptr));

KU_FLOATING_METHOD InitializeWithConf(DeviceCtx* ctx,
                                      const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob) {
  BEFORE_CPU_INITIALIZE();
  // synchronous initialize the host blob
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(
      nullptr, initializer_conf, random_seed, host_blob.get());
  AFTER_CPU_INITIALIZE();
}

KU_FLOATING_METHOD InitializeWithConf(DeviceCtx* ctx,
                                      const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob,
                                      const std::string& data_format) {
  BEFORE_CPU_INITIALIZE();
  // synchronous initialize the host blob
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(
      nullptr, initializer_conf, random_seed, host_blob.get(), data_format);
  AFTER_CPU_INITIALIZE();
}
KU_FLOATING_METHOD InitializeWithDir(DeviceCtx* ctx, int32_t part_id,
                                     int32_t part_num,
                                     const std::string& model_dir, Blob* blob,
                                     const std::string& bn_in_op,
                                     int32_t dim_num, int64_t num_in_each_dim) {
  BEFORE_CPU_INITIALIZE();
  KernelUtil<DeviceType::kCPU, T>::InitializeWithDir(
      ctx, part_id, part_num, model_dir, host_blob.get(), bn_in_op, dim_num,
      num_in_each_dim);
  AFTER_CPU_INITIALIZE();
}

template<typename T>
void KernelUtil<DeviceType::kGPU, T,
                typename std::enable_if<IsIntegral<T>::value>::type>::
    Axpy(DeviceCtx* ctx, const int n, const T alpha, const T* x, const int incx,
         T* y, const int incy) {
  AxpyGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
               ctx->cuda_stream()>>>(n, alpha, x, incx, y, incy);
}

#define INSTANTIATE_KERNEL_UTIL(type_cpp, type_proto)                      \
  template struct GpuKernelUtilIf<type_cpp,                                \
                                  KernelUtil<DeviceType::kGPU, type_cpp>>; \
  template struct KernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_KERNEL_UTIL, ARITHMETIC_DATA_TYPE_SEQ);

template<>
__device__ float gpu_atomic_add(float* address, const float val) {
  return atomicAdd(address, val);
}

template<>
__device__ double gpu_atomic_add(double* address, const double val) {
  auto address_as_ull = reinterpret_cast<unsigned long long int*>(address);
  unsigned long long int old = *address_as_ull;
  unsigned long long int assumed;
  do {
    assumed = old;
    old = atomicCAS(address_as_ull, assumed,
                    __double_as_longlong(val + __longlong_as_double(assumed)));
  } while (assumed != old);
  return __longlong_as_double(old);
}

}  // namespace oneflow
