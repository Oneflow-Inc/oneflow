#include "oneflow/core/kernel/accumulate_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void AccumulateKernel<device_type, T>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("one");
  Blob* out_blob = BnInOp2Blob("acc");
  CHECK_EQ(in_blob->data_type(), GetDataType<T>::val);
  CHECK_EQ(out_blob->data_type(), GetDataType<T>::val);
  KernelUtil<device_type, T>::BlasAxpy(ctx, in_blob->shape().elem_cnt(),
                                       static_cast<T>(1.0), in_blob->dptr<T>(),
                                       1, out_blob->mut_dptr<T>(), 1);
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

COMMAND(AddKernelCreator(OperatorConf::kAccumulateConf, DeviceType::kCPU,
                         CreateAccKernel<DeviceType::kCPU>);
        AddKernelCreator(OperatorConf::kAccumulateConf, DeviceType::kGPU,
                         CreateAccKernel<DeviceType::kGPU>));

}  // namespace oneflow
