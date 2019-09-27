#include "oneflow/core/kernel/local_response_normalization_kernel.h"

namespace oneflow {

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kGPU, T>::VirtualKernelInit() {
  const PbRf<int64_t>& shape = GetPbRfFromPbMessage<int64_t>(
      GetValFromPbMessage<const PbMessage&>(this->kernel_conf().local_response_normalization_conf(),
                                            "batch"),
      "dim");
  batch_desc_.reset(new CudnnTensorDesc(CUDNN_TENSOR_NCHW, GetDataType<T>::value, shape.Get(0),
                                        shape.Get(1), shape.Get(2), shape.Get(3)));
  const LocalResponseNormalizationOpConf& op_conf =
      this->op_conf().local_response_normalization_conf();
  normalize_desc_.reset(
      new CudnnLRNDesc(op_conf.depth_radius(), op_conf.alpha(), op_conf.beta(), op_conf.bias()));
}

template<typename T>
void LocalResponseNormalizationKernel<DeviceType::kGPU, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CudaCheck(cudnnLRNCrossChannelForward(
      ctx.device_ctx->cudnn_handle(), normalize_desc_->Get(), CUDNN_LRN_CROSS_CHANNEL_DIM1,
      CudnnSPOnePtr<T>(), batch_desc_->Get(), BnInOp2Blob("in")->dptr(), CudnnSPZeroPtr<T>(),
      batch_desc_->Get(), BnInOp2Blob("out")->mut_dptr()));
}

#define INSTANTIATE_LOCAL_RESPONSE_NORMALIZATION_KERNEL(type_cpp, type_proto) \
  template class LocalResponseNormalizationKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LOCAL_RESPONSE_NORMALIZATION_KERNEL, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
