#include "oneflow/core/kernel/normalization_kernel.h"

namespace oneflow {

template<>
void NormalizationKernel<DeviceType::kGPU, float>::NormalizationCudnnForward(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const float* in = BnInOp2Blob("in")->dptr<float>();
  const float* gamma = BnInOp2Blob("gamma")->dptr<float>();
  const float* beta = BnInOp2Blob("beta")->dptr<float>();
  float* out = BnInOp2Blob("out")->mut_dptr<float>();
  float* moving_mean = BnInOp2Blob("moving_mean")->mut_dptr<float>();
  float* moving_variance = BnInOp2Blob("moving_variance")->mut_dptr<float>();
  double epsilon = this->op_conf().normalization_conf().epsilon();
  if (Global<JobDesc>::Get()->IsTrain()) {
    InitMovingMeanAndMovingVariance(ctx, BnInOp2Blob, false);
    double momentum = this->op_conf().normalization_conf().momentum();
    CudaCheck(cudnnBatchNormalizationForwardTraining(
        ctx.device_ctx->cudnn_handle(),
        normalization_ctx_->cudnn_batch_norm_mode(), OnePtr<float>::value,
        ZeroPtr<float>::value, normalization_ctx_->cudnn_in_tensor_desc(), in,
        normalization_ctx_->cudnn_in_tensor_desc(), out,
        normalization_ctx_->cudnn_param_tensor_desc(), gamma, beta,
        1 - momentum, moving_mean, moving_variance, epsilon,
        BnInOp2Blob("cache_mean_for_cudnn_bw")->mut_dptr<float>(),
        BnInOp2Blob("cache_inv_variance_for_cudnn_bw")->mut_dptr<float>()));
  } else {
    CudaCheck(cudnnBatchNormalizationForwardInference(
        ctx.device_ctx->cudnn_handle(), CUDNN_BATCHNORM_SPATIAL,
        OnePtr<float>::value, ZeroPtr<float>::value,
        normalization_ctx_->cudnn_in_tensor_desc(), in,
        normalization_ctx_->cudnn_in_tensor_desc(), out,
        normalization_ctx_->cudnn_param_tensor_desc(), gamma, beta, moving_mean,
        moving_variance, epsilon));
  }
}
template<>
void NormalizationKernel<DeviceType::kGPU, float>::NormalizationCudnnBackward(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  const cudnnTensorDescriptor_t& in_desc =
      normalization_ctx_->cudnn_in_tensor_desc();
  CudaCheck(cudnnBatchNormalizationBackward(
      ctx.device_ctx->cudnn_handle(),
      normalization_ctx_->cudnn_batch_norm_mode(), OnePtr<float>::value,
      ZeroPtr<float>::value, OnePtr<float>::value, ZeroPtr<float>::value,
      in_desc, BnInOp2Blob("in")->dptr<float>(), in_desc,
      BnInOp2Blob(GenDiffBn("out"))->dptr<float>(), in_desc,
      BnInOp2Blob(GenDiffBn("in"))->mut_dptr<float>(),
      normalization_ctx_->cudnn_param_tensor_desc(),
      BnInOp2Blob("gamma")->dptr<float>(),
      BnInOp2Blob(GenDiffBn("gamma"))->mut_dptr<float>(),
      BnInOp2Blob(GenDiffBn("beta"))->mut_dptr<float>(),
      static_cast<double>(this->op_conf().normalization_conf().epsilon()),
      BnInOp2Blob("cache_mean_for_cudnn_bw")->dptr<float>(),
      BnInOp2Blob("cache_inv_variance_for_cudnn_bw")->dptr<float>()));
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::NormalizationCudnnForward(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  UNIMPLEMENTED();
}

template<DeviceType device_type, typename T>
void NormalizationKernel<device_type, T>::NormalizationCudnnBackward(
    const KernelCtx& ctx,
    const std::function<Blob*(const std::string&)>& BnInOp2Blob) const {
  UNIMPLEMENTED();
}

#define MAKE_NORMALIZATION_KERNEL_ENTRY(device_type, data_type) \
  template class NormalizationKernel<device_type, OF_PP_PAIR_FIRST(data_type)>;

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_NORMALIZATION_KERNEL_ENTRY,
                                 DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)

}  // namespace oneflow
