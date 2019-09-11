#include "oneflow/core/kernel/unsorted_segment_sum_kernel.h"
#include "oneflow/core/kernel/batch_gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& UnsortedSegmentSumKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().unsorted_segment_sum_conf();
}

template<DeviceType device_type, typename T>
void UnsortedSegmentSumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  BatchGatherKernelUtil<device_type, T>::Backward(ctx.device_ctx, BnInOp2Blob("data"),
                                                  BnInOp2Blob("segment_ids"), BnInOp2Blob("out"));
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kUnsortedSegmentSumConf, UnsortedSegmentSumKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
