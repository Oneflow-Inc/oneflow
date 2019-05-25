#include "oneflow/core/kernel/conv_data_grad_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ConvDataGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  ConvDataGradKernelUtil<device_type, T>::Compute(
      ctx.device_ctx, this->kernel_conf().conv_data_grad_conf(),
      this->op_conf().conv_data_grad_conf().conv_conf(), BnInOp2Blob("dy"), BnInOp2Blob("filter"),
      BnInOp2Blob("dx"), BnInOp2Blob("buf"));
}

template<DeviceType device_type, typename T>
const PbMessage& ConvDataGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().conv_data_grad_conf();
}

template<typename T>
struct ConvDataGradKernelUtil<DeviceType::kCPU, T> final {
  static void Compute(DeviceCtx* ctx, const ConvDataGradKernelConf& kernel_conf,
                      const ConvConf& conf, const Blob* dy, const Blob* filter, Blob* dx,
                      Blob* buf) {
    UNIMPLEMENTED();
  }
};

ADD_GPU_HALF_KERNEL_CREATOR(OperatorConf::kConvDataGradConf, ConvDataGradKernel,
                            FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
