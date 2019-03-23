#include "oneflow/core/kernel/layer_norm_grad_kernel.h"
#include "oneflow/core/kernel/layer_norm_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void LayerNormGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const LayerNormGradOpConf& conf = this->op_conf().layer_norm_grad_conf();
  const Blob* dy = BnInOp2Blob("dy");
  const Blob* x = BnInOp2Blob("x");
  const Blob* mean = BnInOp2Blob("mean");
  const Blob* inv_variance = BnInOp2Blob("inv_variance");
  if (mean || inv_variance) {
    CHECK_NOTNULL(mean);
    CHECK_NOTNULL(inv_variance);
  }
  const Blob* bn_scale = BnInOp2Blob("cudnn_bn_scale_ones");
  Blob* dx = BnInOp2Blob("dx");
  Blob* bn_scale_diff = BnInOp2Blob("cudnn_bn_scale_diff_buf");
  Blob* bn_bias_diff = BnInOp2Blob("cudnn_bn_bias_diff_buf");
  LayerNormKernelUtil<device_type, T>::NormalizeBackward(ctx.device_ctx, x, bn_scale, mean,
                                                         inv_variance, dy, conf.epsilon(), dx,
                                                         bn_scale_diff, bn_bias_diff);
}

template<DeviceType device_type, typename T>
void LayerNormGradKernel<device_type, T>::InitConstBufBlobs(
    DeviceCtx* ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  InitializerConf ones_initializer;
  ones_initializer.mutable_constant_conf()->set_value(1.0);
  KernelUtil<device_type, T>::InitializeWithConf(ctx, ones_initializer, 0,
                                                 BnInOp2Blob("cudnn_bn_scale_ones"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kLayerNormGradConf, LayerNormGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
