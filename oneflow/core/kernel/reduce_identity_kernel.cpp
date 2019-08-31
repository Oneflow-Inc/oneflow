#include "oneflow/core/kernel/reduce_identity_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReduceIdentityKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  CHECK_EQ(out_blob->ByteSizeOfDataContentField(), in_blob->ByteSizeOfDataContentField());
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr(), in_blob->dptr(),
                      out_blob->ByteSizeOfDataContentField());
}

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kReduceIdentityConf, DeviceType::kCPU,
                            ReduceIdentityKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kReduceIdentityConf, DeviceType::kGPU,
                            ReduceIdentityKernel<DeviceType::kGPU>);

}  // namespace oneflow
