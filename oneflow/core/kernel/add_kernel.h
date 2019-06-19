#ifndef ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_
#define ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_

#include "oneflow/core/kernel/kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
class AddKernel final
    : public std::conditional<IsFloating<T>::value, KernelIfWithActivation<device_type, T>,
                              KernelIf<device_type>>::type {
 public:
  OF_DISALLOW_COPY_AND_MOVE(AddKernel);
  AddKernel() = default;
  ~AddKernel() = default;

 private:
  void ForwardDataContent(const KernelCtx& ctx,
                          std::function<Blob*(const std::string&)> BnInOp2Blob) const override;

  const PbMessage& GetCustomizedOpConf() const override;

  decltype(make_tuple_from_sequence<7>()) tp_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ADD_KERNEL_H_
