#include "oneflow/core/kernel/conv_bias_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvBiasGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ConvBiasGradKernelUtil<device_type, T>::Compute(
      ctx.device_ctx, this->op_conf().conv_bias_grad_conf().data_format(), BnInOp2Blob("dy"),
      BnInOp2Blob("bias_diff"));
}

template<DeviceType device_type, typename T>
const PbMessage& ConvBiasGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().conv_bias_grad_conf();
}

template<typename T>
struct ConvBiasGradKernelUtil<DeviceType::kCPU, T> final {
  static void Compute(DeviceCtx* ctx, const std::string& format, const Blob* dy, Blob* bias_diff) {
    UNIMPLEMENTED();
  }
};

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kConvBiasGradConf, ConvBiasGradKernel);
REGISTER_KERNEL_HELPER_GPU_HALF(OperatorConf::kConvBiasGradConf, ConvBiasGradKernel);

}  // namespace oneflow
