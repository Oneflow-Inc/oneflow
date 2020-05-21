#include "oneflow/core/kernel/pack_kernel.h"
#include "oneflow/core/kernel/pack_kernel_util.h"

namespace oneflow {

template<DeviceType device_type>
void PackKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto* res = static_cast<std::pair<size_t, size_t>*>(ctx.other);
  size_t in_index = res->first;
  size_t total_pack_num = res->second;
  PackKernelUtil<device_type>::Pack(ctx.device_ctx, in_index, total_pack_num, BnInOp2Blob("in"),
                                    BnInOp2Blob("out"));
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kPackConf, PackKernel);

}  // namespace oneflow
