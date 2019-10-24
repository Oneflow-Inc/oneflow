#include "oneflow/core/kernel/l2_normalize_kernel.h"
#include "oneflow/core/kernel/l2_normalize_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void L2NormalizeKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  L2NormalizeKernelUtil<device_type, T>::Forward(
      ctx.device_ctx, this->op_conf().l2_normalize_conf().axis(),
      this->op_conf().l2_normalize_conf().epsilon(), BnInOp2Blob("in"), BnInOp2Blob("square_x_sum"),
      BnInOp2Blob("out"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kL2NormalizeConf, L2NormalizeKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
