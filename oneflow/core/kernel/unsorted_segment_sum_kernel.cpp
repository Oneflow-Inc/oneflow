#include "oneflow/core/kernel/unsorted_segment_sum_kernel.h"
#include "oneflow/core/kernel/gather_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T>
const PbMessage& UnsortedSegmentSumKernel<device_type, T>::GetCustomizedOpConf() const {
  return this->op_conf().unsorted_segment_sum_conf();
}

template<DeviceType device_type, typename T>
void UnsortedSegmentSumKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* segment_ids = BnInOp2Blob("segment_ids");
  const Blob* data = BnInOp2Blob("data");
  Blob* out = BnInOp2Blob("out");
  Memset<device_type>(ctx.device_ctx, out->mut_dptr<T>(), 0, out->ByteSizeOfDataContentField());
  GatherKernelUtil<device_type, T>::Backward(
      ctx.device_ctx, segment_ids, data, this->op_conf().unsorted_segment_sum_conf().axis(), out);
}

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kUnsortedSegmentSumConf,
                                    UnsortedSegmentSumKernel);
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700
REGISTER_KERNEL_HELPER_GPU_HALF(OperatorConf::kUnsortedSegmentSumConf, UnsortedSegmentSumKernel);
#endif

}  // namespace oneflow
