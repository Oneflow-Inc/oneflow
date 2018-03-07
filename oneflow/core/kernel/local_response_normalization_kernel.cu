#include "oneflow/core/kernel/local_response_normalization_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LocalResponseNormalizationKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CudaCheck(cudnnLRNCrossChannelForward(
      ctx.device_ctx->cudnn_handle(), normalize_desc_->Get(),
      CUDNN_LRN_CROSS_CHANNEL_DIM1, CudnnDataType<T>::one, batch_desc_->Get(),
      BnInOp2Blob("in")->dptr(), CudnnDataType<T>::zero, batch_desc_->Get(),
      BnInOp2Blob("out")->mut_dptr()));
}

template<DeviceType device_type, typename T>
void LocalResponseNormalizationKernel<device_type, T>::BackwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  CudaCheck(cudnnLRNCrossChannelBackward(
      ctx.device_ctx->cudnn_handle(), normalize_desc_->Get(),
      CUDNN_LRN_CROSS_CHANNEL_DIM1, CudnnDataType<T>::one, batch_desc_->Get(),
      BnInOp2Blob("out")->dptr(), batch_desc_->Get(),
      BnInOp2Blob("out_diff")->dptr(), batch_desc_->Get(),
      BnInOp2Blob("in")->dptr(), CudnnDataType<T>::zero, batch_desc_->Get(),
      BnInOp2Blob("in_diff")->mut_dptr()));
}

#define INSTANTIATE_LOCAL_RESPONSE_NORMALIZATION_KERNEL(type_cpp, type_proto) \
  template class LocalResponseNormalizationKernel<DeviceType::kGPU, type_cpp>;
OF_PP_FOR_EACH_TUPLE(INSTANTIATE_LOCAL_RESPONSE_NORMALIZATION_KERNEL,
                     FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
