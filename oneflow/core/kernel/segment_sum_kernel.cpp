#include "oneflow/core/kernel/segment_sum_kernel.h"
#include "oneflow/core/kernel/segment_kernel_util.h"

namespace oneflow {

template <DeviceType device_type, typename T>
const PbMessage& SegmentSumKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().segment_sum_conf();
}

template <DeviceType device_type, typename T>
void SegmentSumKernel<device_type, T>::ForwardDataContent(const KernelCtx& ctx,
                                    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in = BnInOp2Blob("in");
  const Blob* segment_ids = BnInOp2Blob("segment_ids");
  Blob* out = BnInOp2Blob("out"); 
  Memset<device_type>(ctx.device_ctx, out->mut_dptr(), 0, out->ByteSizeOfDataContentField());
  SegmentKernelUtil<device_type, float, int32_t>::SegmentSumForward(ctx.device_ctx, in, segment_ids, out);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kSegmentSumConf, SegmentSumKernel, FLOATING_DATA_TYPE_SEQ);

} // namespace oneflow
