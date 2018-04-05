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

KU_FLOATING_METHOD InitializeWithConf(DeviceCtx* ctx,
                                      const InitializerConf& initializer_conf,
                                      uint32_t random_seed, Blob* blob) {
  // create temporary host blob store initializer result
  BlobDesc blob_desc = BlobDesc(blob->blob_desc());
  char* host_raw_dptr = nullptr;
  CudaCheck(cudaMallocHost(&host_raw_dptr, blob->TotalByteSize()));
  std::unique_ptr<Blob> host_blob;
  host_blob.reset(
      NewBlob(nullptr, &blob_desc, host_raw_dptr, nullptr, DeviceType::kGPU));
  // synchronous initialize the host blob
  KernelUtil<DeviceType::kCPU, T>::InitializeWithConf(
      nullptr, initializer_conf, random_seed, host_blob.get());
  // asynchronous copy to device
  Memcpy<DeviceType::kGPU>(ctx, blob->mut_dptr(), host_blob->dptr(),
                           blob->ByteSizeOfDataContentField(),
                           cudaMemcpyHostToDevice);
  cudaStreamSynchronize(ctx->cuda_stream());
  CudaCheck(cudaFreeHost(host_raw_dptr));
}

KU_FLOATING_METHOD InitializeWithDir(DeviceCtx* ctx, int32_t part_id,
                                     int32_t part_num,
                                     const std::string& model_dir, Blob* blob,
                                     const std::string& bn_in_op,
                                     int32_t dim_num, int64_t num_in_each_dim) {
  BlobDesc blob_desc = BlobDesc(blob->blob_desc());
  char* host_raw_dptr = nullptr;
  CudaCheck(cudaMallocHost(&host_raw_dptr, blob->TotalByteSize()));
  std::unique_ptr<Blob> host_blob;
  host_blob.reset(
      NewBlob(nullptr, &blob_desc, host_raw_dptr, nullptr, DeviceType::kGPU));
  KernelUtil<DeviceType::kCPU, T>::InitializeWithDir(
      ctx, part_id, part_num, model_dir, host_blob.get(), bn_in_op, dim_num,
      num_in_each_dim);

  Memcpy<DeviceType::kGPU>(ctx, blob->mut_dptr(), host_blob->dptr(),
                           blob->ByteSizeOfDataContentField(),
                           cudaMemcpyHostToDevice);
  cudaStreamSynchronize(ctx->cuda_stream());
  CudaCheck(cudaFreeHost(host_raw_dptr));
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
