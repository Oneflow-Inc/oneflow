#include "oneflow/core/kernel/tanh_kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void TanHKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  NewKernelUtil<device_type>::TanH(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                   BnInOp2Blob("out")->mut_dptr<T>());
}

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kTanhConf, TanHKernel);
REGISTER_KERNEL_HELPER_GPU_HALF(OperatorConf::kTanhConf, TanHKernel);

}  // namespace oneflow
