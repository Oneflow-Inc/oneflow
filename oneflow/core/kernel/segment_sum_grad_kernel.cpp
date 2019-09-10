#include "oneflow/core/kernel/segment_sum_grad_kernel.h"
#include "oneflow/core/kernel/segment_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& SegmentSumGradKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().segment_sum_grad_conf();
}

template<DeviceType device_type, typename T>
void SegmentSumGradKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff = BnInOp2Blob("out_diff");
  const Blob* segment_ids = BnInOp2Blob("segment_ids");
  Blob* in_diff = BnInOp2Blob("in_diff");
  Memset<device_type>(ctx.device_ctx, in_diff->mut_dptr(), 0,
                      in_diff->ByteSizeOfDataContentField());
  SegmentKernelUtil<device_type, float, int32_t>::SegmentSumBackward(ctx.device_ctx, out_diff,
                                                                     segment_ids, in_diff);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSegmentSumGradConf, SegmentSumGradKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
