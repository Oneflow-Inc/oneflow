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

namespace {

Kernel* CreateUnsortedSegmentSumKernel(const KernelConf& kernel_conf) {
  static const HashMap<std::string, std::function<Kernel*()>> creators = {
    OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(MAKE_KERNEL_CREATOR_ENTRY, (UnsortedSegmentSumKernel),
                                     DEVICE_TYPE_SEQ, FLOATING_DATA_TYPE_SEQ)
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 700 && CUDA_VERSION >= 10000
        MAKE_KERNEL_CREATOR_ENTRY(UnsortedSegmentSumKernel, DeviceType::kGPU,
                                  (float16, DataType::kFloat16))
#endif
  };
  return creators.at(
      GetHashKey(kernel_conf.op_attribute().op_conf().device_type(), kernel_conf.data_type()))();
}

REGISTER_KERNEL_CREATOR(OperatorConf::kUnsortedSegmentSumConf, CreateUnsortedSegmentSumKernel);
}  // namespace

}  // namespace oneflow
