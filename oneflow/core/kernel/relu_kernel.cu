#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void ReluForwardGpu(const int64_t n, const T* in, T* out) {
  CUDA_1D_KERNEL_LOOP(i, n) { out[i] = in[i] > 0 ? in[i] : 0; }
}

template<typename T>
__global__ void ReluBackwardGpu(const int64_t n, const T* out_diff, const T* in,
                                T* in_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) { in_diff[i] = in[i] > 0 ? out_diff[i] : 0; }
}

}  // namespace

template<typename T>
class ReluKernelUtil<DeviceType::kGPU, T> final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ReluKernelUtil);
  ReluKernelUtil() = delete;

  static void Forward(const KernelCtx& ctx, const int64_t n, const T* in,
                      T* out) {
    ReluForwardGpu<T><<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
                        ctx.device_ctx->cuda_stream()>>>(n, in, out);
  }

  static void Backward(const KernelCtx& ctx, const int64_t n, const T* out_diff,
                       const T* in, T* in_diff) {
    ReluBackwardGpu<T>
        <<<BlocksNum4ThreadsNum(n), kCudaThreadsNumPerBlock, 0,
           ctx.device_ctx->cuda_stream()>>>(n, out_diff, in, in_diff);
  }
};

#ifdef USE_CUDNN
template<typename T>
CudnnReluKernel<T>::CudnnReluKernel() {
  CudaCheck(cudnnCreateTensorDescriptor(&in_desc_));
  CudaCheck(cudnnCreateTensorDescriptor(&out_desc_));
  CudaCheck(cudnnCreateActivationDescriptor(&activ_desc_));
  CudaCheck(cudnnSetActivationDescriptor(activ_desc_, CUDNN_ACTIVATION_RELU,
                                         CUDNN_PROPAGATE_NAN, 0.0));
}

template<typename T>
CudnnReluKernel<T>::~CudnnReluKernel() {
  CudaCheck(cudnnDestroyTensorDescriptor(in_desc_));
  CudaCheck(cudnnDestroyTensorDescriptor(out_desc_));
  CudaCheck(cudnnDestroyActivationDescriptor(activ_desc_));
}

template<typename T>
void CudnnReluKernel<T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");

  int64_t in_height =
      in_blob->shape().NumAxes() < 4 ? 1 : in_blob->shape().At(2);
  int64_t in_width =
      in_blob->shape().NumAxes() < 4 ? 1 : in_blob->shape().At(3);
  int64_t out_height =
      out_blob->shape().NumAxes() < 4 ? 1 : out_blob->shape().At(2);
  int64_t out_width =
      out_blob->shape().NumAxes() < 4 ? 1 : out_blob->shape().At(3);

  CudaCheck(cudnnSetTensor4dDescriptor(
      in_desc_, CUDNN_TENSOR_NCHW, CudnnDataType<T>::type,
      in_blob->shape().At(0), in_blob->shape().At(1), in_height, in_width));
  CudaCheck(cudnnSetTensor4dDescriptor(
      out_desc_, CUDNN_TENSOR_NCHW, CudnnDataType<T>::type,
      out_blob->shape().At(0), out_blob->shape().At(1), out_height, out_width));

  CudaCheck(cudnnActivationForward(ctx.device_ctx->cudnn_handle(), activ_desc_,
                                   CudnnDataType<T>::one, in_desc_,
                                   in_blob->dptr<T>(), CudnnDataType<T>::zero,
                                   out_desc_, out_blob->mut_dptr<T>()));
}

template<typename T>
void CudnnReluKernel<T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const Blob* out_blob = BnInOp2Blob("out");
  const Blob* out_diff_blob = BnInOp2Blob("out_diff");
  Blob* in_diff_blob = BnInOp2Blob("in_diff");

  Memset<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_dptr(), 0,
                           in_diff_blob->ByteSizeOfDataContentField());

  CudaCheck(cudnnActivationBackward(
      ctx.device_ctx->cudnn_handle(), activ_desc_, CudnnDataType<T>::one,
      out_desc_, out_blob->dptr<T>(), out_desc_, out_diff_blob->dptr<T>(),
      in_desc_, in_blob->dptr<T>(), CudnnDataType<T>::zero, in_desc_,
      in_diff_blob->mut_dptr<T>()));
}

#define INSTANTIATE_RELU_KERNEL(type_cpp, type_proto) \
  template class CudnnReluKernel<type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_RELU_KERNEL,
                     FLOATING_DATA_TYPE_SEQ)  // TODO(shiyuan): cudnn does not
                                              // support "signed char"
#endif                                        // USE_CUDNN

#define INSTANTIATE_RELU_KERNEL_UTIL(type_cpp, type_proto) \
  template class ReluKernelUtil<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_RELU_KERNEL_UTIL,
                     FLOATING_DATA_TYPE_SEQ SIGNED_INT_DATA_TYPE_SEQ)

}  // namespace oneflow
