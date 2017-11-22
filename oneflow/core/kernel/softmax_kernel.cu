#include "oneflow/core/kernel/kernel_manager.h"
#include "oneflow/core/kernel/softmax_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void SoftmaxForwardMaxGpu(const int64_t n, const int64_t w,
                                     const T* out, T* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T max_value = out[i * w];
    for (int64_t j = 0; j < w; ++j) {
      max_value = max_value > out[i * w + j] ? max_value : out[i * w + j];
    }
    tmp[i] = max_value;
  }
}

template<typename T>
__global__ void SoftmaxForwardSumGpu(const int64_t n, const int64_t w,
                                     const T* out, T* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T sum_value = 0;
    for (int64_t j = 0; j < w; ++j) { sum_value += out[i * w + j]; }
    tmp[i] = sum_value;
  }
}

template<typename T>
__global__ void SoftmaxSubGpu(const int64_t n, const int64_t w, T* matrix,
                              const T* vector) {
  CUDA_1D_KERNEL_LOOP(i, n * w) { matrix[i] -= vector[i / w]; }
}

template<typename T>
__global__ void SoftmaxBackwardDotGpu(const int64_t n, const int64_t w,
                                      const T* out, const T* out_diff, T* tmp) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    T dot_result = 0;
    for (int64_t j = 0; j < w; ++j) {
      dot_result += out[i * w + j] * out_diff[i * w + j];
    }
    tmp[i] = dot_result;
  }
}

}  // namespace

template<typename T>
class SoftmaxKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(SoftmaxKernelUtil);
  SoftmaxKernelUtil() = delete;

  static void ForwardMax(DeviceCtx* ctx, const int64_t n, const int64_t w,
                         const T* out, T* tmp) {
    SoftmaxForwardMaxGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock,
                              0, ctx->cuda_stream()>>>(n, w, out, tmp);
  }

  static void ForwardSum(DeviceCtx* ctx, const int64_t n, const int64_t w,
                         const T* out, T* tmp) {
    SoftmaxForwardSumGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock,
                              0, ctx->cuda_stream()>>>(n, w, out, tmp);
  }

  static void Sub(DeviceCtx* ctx, const int64_t n, const int64_t w, T* matrix,
                  const T* vector) {
    SoftmaxSubGpu<T><<<BlocksNum4ThreadsNum(n * w), kCudaThreadsNumPerBlock, 0,
                       ctx->cuda_stream()>>>(n, w, matrix, vector);
  }

  static void BackwardDot(DeviceCtx* ctx, const int64_t n, const int64_t w,
                          const T* out, const T* out_diff, T* tmp) {
    SoftmaxBackwardDotGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx->cuda_stream()>>>(n, w, out, out_diff, tmp);
  }
};

template<typename T>
CudnnSoftmaxKernel<DeviceType::kGPU, T>::CudnnSoftmaxKernel() {
  CudaCheck(cudnnCreateTensorDescriptor(&in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc_));
}

template<typename T>
CudnnSoftmaxKernel<DeviceType::kGPU, T>::~CudnnSoftmaxKernel() {
  CudaCheck(cudnnDestroyTensorDescriptor(in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc_));
}

template<typename T>
void CudnnSoftmaxKernel<DeviceType::kGPU, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(op()->SoleIbn());
  Blob* out_blob = BnInOp2Blob(op()->SoleObn());
  CopyDataIdFromSoleIbToAllObIfNeed<DeviceType::kGPU>(ctx, BnInOp2Blob);

  CudaCheck(cudnnSetTensor4dDescriptor(
      in_desc_, CUDNN_TENSOR_NCHW, cudnn::DataType<T>::type,
      in_blob->shape().At(0), in_blob->shape().At(1), in_blob->shape().At(2),
      in_blob->shape().At(3)));
  CudaCheck(cudnnSetTensor4dDescriptor(
      out_desc_, CUDNN_TENSOR_NCHW, cudnn::DataType<T>::type,
      out_blob->shape().At(0), out_blob->shape().At(1), out_blob->shape().At(2),
      out_blob->shape().At(3)));

  CudaCheck(cudnnSoftmaxForward(
      ctx.device_ctx->cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL, cudnn::DataType<T>::one, in_desc_,
      in_blob->dptr<T>(), cudnn::DataType<T>::zero, out_desc_,
      out_blob->mut_dptr<T>()));
}

template<typename T>
void CudnnSoftmaxKernel<DeviceType::kGPU, T>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_blob = BnInOp2Blob(op()->SoleObn());
  const Blob* out_diff_blob = BnInOp2Blob(op()->SoleOdbn());
  Blob* in_diff_blob = BnInOp2Blob(op()->SoleIdbn());
  Memset<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataField());

  CudaCheck(cudnnSoftmaxBackward(
      ctx.device_ctx->cudnn_handle(), CUDNN_SOFTMAX_ACCURATE,
      CUDNN_SOFTMAX_MODE_CHANNEL, cudnn::DataType<T>::one, out_desc_,
      out_blob->dptr<T>(), out_desc_, out_diff_blob->dptr<T>(),
      cudnn::DataType<T>::zero, in_desc_, in_diff_blob->mut_dptr<T>()));
}

#ifdef USE_CUDNN
#define INSTANTIATE_SOFTMAX_KERNEL(type_cpp, type_proto) \
  template class CudnnSoftmaxKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_KERNEL, FLOATING_DATA_TYPE_SEQ)
#endif
#define INSTANTIATE_SOFTMAX_KERNEL_UTIL(type_cpp, type_proto) \
  template class SoftmaxKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_SOFTMAX_KERNEL_UTIL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
