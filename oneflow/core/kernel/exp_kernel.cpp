#include "oneflow/core/kernel/exp_kernel.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ExpKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  KernelUtil<device_type, T>::Exp(ctx.device_ctx, in_blob->shape().elem_cnt(), in_blob->dptr<T>(),
                                  BnInOp2Blob("out")->mut_dptr<T>());
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kExpConf, ExpKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
