#include "oneflow/core/kernel/relu_kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  NewKernelUtil<device_type>::Relu(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                   BnInOp2Blob("out")->mut_dptr<T>());
}

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kReluConf, ReluKernel);
REGISTER_KERNEL_HELPER_GPU_HALF(OperatorConf::kReluConf, ReluKernel);

}  // namespace oneflow
