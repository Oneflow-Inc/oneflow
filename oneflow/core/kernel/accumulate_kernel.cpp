#include "oneflow/core/kernel/accumulate_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void AccumulateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("one");
  Blob* out_blob = BnInOp2Blob("acc");
  KernelUtil<device_type, T>::BlasAxpy(
      ctx.device_ctx, in_blob->shape().elem_cnt(), static_cast<T>(1.0),
      in_blob->dptr<T>(), 1, out_blob->mut_dptr<T>(), 1);
}

Kernel* CreateAccumulateKernel(const OpContext& op_ctx) {
  static const HashMap<std::string, std::function<Kernel*()>> creator = {
#define ACCUMULATE_KERNEL_ENTRY(device_type, data_type_pair)          \
  {GetHashKey(device_type, OF_PP_PAIR_SECOND(data_type_pair)), []() { \
     return new AccumulateKernel<device_type,                         \
                                 OF_PP_PAIR_FIRST(data_type_pair)>;   \
   }},
      OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(ACCUMULATE_KERNEL_ENTRY, DEVICE_TYPE_SEQ,
                                       FLOATING_DATA_TYPE_SEQ)};

  return creator.at(GetHashKey(op_ctx.device_type(),
                               JobDesc::Singleton()->default_data_type()))();
}

COMMAND(AddKernelCreator(OperatorConf::kAccumulateConf,
                         CreateAccumulateKernel));

}  // namespace oneflow
