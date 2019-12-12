#include "oneflow/core/kernel/conv_filter_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvFilterGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const bool enable_true_half = this->job_desc().enable_true_half_config_when_conv();
  ConvFilterGradKernelUtil<device_type, T>::Compute(
      ctx.device_ctx, this->kernel_conf().conv_filter_grad_conf(),
      this->op_conf().conv_filter_grad_conf().conv_conf(), BnInOp2Blob("x"), BnInOp2Blob("dy"),
      BnInOp2Blob("filter_diff"), BnInOp2Blob("buf"), enable_true_half);
}

template<DeviceType device_type, typename T>
const PbMessage& ConvFilterGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().conv_filter_grad_conf();
}

template<typename T>
struct ConvFilterGradKernelUtil<DeviceType::kCPU, T> final {
  static void Compute(DeviceCtx* ctx, const ConvFilterGradKernelConf& kernel_conf,
                      const ConvConf& conf, const Blob* x, const Blob* dy, Blob* filter_diff,
                      Blob* buf, const bool enable_true_half) {
    UNIMPLEMENTED();
  }
};

ADD_DEFAULT_KERNEL_CREATOR_WITH_GPU_HALF(OperatorConf::kConvFilterGradConf, ConvFilterGradKernel,
                                         FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
