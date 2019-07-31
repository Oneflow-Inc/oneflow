#include "oneflow/core/kernel/gelu_kernel.h"
#include "oneflow/core/kernel/gelu_grad_kernel.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void GeluGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* x_blob = BnInOp2Blob("x");
  GeluKernelUtil<device_type, T>::GeluBackward(ctx.device_ctx, x_blob->static_shape().elem_cnt(),
                                               x_blob->dptr<T>(), BnInOp2Blob("dy")->dptr<T>(),
                                               BnInOp2Blob("dx")->mut_dptr<T>());
}

template<DeviceType device_type, typename T>
const PbMessage& GeluGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().gelu_grad_conf();
}

#define REGISTER_GELU_GRAD_KERNEL(dev, dtype)                                    \
  REGISTER_KERNEL_WITH_DEVICE_AND_DTYPE(OperatorConf::kGeluGradConf, dev, dtype, \
                                        GeluGradKernel<dev, dtype>)

REGISTER_GELU_GRAD_KERNEL(DeviceType::kGPU, float);
REGISTER_GELU_GRAD_KERNEL(DeviceType::kGPU, double);
REGISTER_GELU_GRAD_KERNEL(DeviceType::kCPU, float);
REGISTER_GELU_GRAD_KERNEL(DeviceType::kCPU, double);

}  // namespace oneflow
