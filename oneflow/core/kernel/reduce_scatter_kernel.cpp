#include "oneflow/core/kernel/relu_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReluKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  KernelUtil<device_type, T>::Relu(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                   BnInOp2Blob("out")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReluConf, ReluKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
