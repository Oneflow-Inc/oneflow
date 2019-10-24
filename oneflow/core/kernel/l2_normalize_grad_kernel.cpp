#include "oneflow/core/kernel/l2_normalize_grad_kernel.h"
#include "oneflow/core/kernel/l2_normalize_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void L2NormalizeGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, this->op_conf().l2_normalize_grad_conf().axis(),
      this->op_conf().l2_normalize_grad_conf().epsilon(), BnInOp2Blob("y"), BnInOp2Blob("dy"),
      BnInOp2Blob("square_x_sum"), BnInOp2Blob("dx"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kL2NormalizeGradConf, L2NormalizeGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
