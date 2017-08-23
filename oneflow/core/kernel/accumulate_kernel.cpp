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

namespace {
template<DeviceType device_type>
Kernel* CreateAccKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define MACRO_PAIR(type_cpp, type_proto) \
  {type_proto, []() { return new AccumulateKernel<device_type, type_cpp>; }},
      FLOATING_DATA_TYPE_PAIR()
#undef MACRO_PAIR
  };
  return data_type2creator.at(op_conf.accumulate_conf().data_type())();
}
}  // namespace

REIGSTER_TEMPLATE_KERNEL_CREATOR(OperatorConf::kAccumulateConf,
                                 CreateAccKernel);

}  // namespace oneflow
