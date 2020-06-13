#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/high_order_bool.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

namespace user_op {

class KernelRegContext;

class KernelRegBoolFunctor final : public hob::BoolFunctor<KernelRegContext> {
 public:
  OF_DISALLOW_COPY_AND_MOVE(KernelRegBoolFunctor)
  KernelRegBoolFunctor() = delete;
  KernelRegBoolFunctor(std::string debug_str, std::function<bool(const KernelRegContext&)> match_fn)
      : debug_str_(debug_str), match_fn_(match_fn) {}
  ~KernelRegBoolFunctor() {}

  bool operator()(const KernelRegContext& ctx) const override;
  std::string DebugStr(const KernelRegContext& ctx, bool display_result) const override;

 private:
  std::string debug_str_;
  std::function<bool(const KernelRegContext&)> match_fn_;
};

hob::BoolFunctorPtr<KernelRegContext> HobDeviceTypeEq(DeviceType device_type);

template<DeviceType device_type>
hob::BoolFunctorPtr<KernelRegContext> HobDeviceTypeEq() {
  return HobDeviceTypeEq(device_type);
}

hob::BoolFunctorPtr<KernelRegContext> HobDataTypeEq(const std::string& tensor_name, int tensor_idx,
                                                    DataType date_type);

template<typename dtype>
hob::BoolFunctorPtr<KernelRegContext> HobDataTypeEq(const std::string& tensor_name,
                                                    int tensor_idx) {
  return HobDataTypeEq(tensor_name, tensor_idx, GetDataType<dtype>::value);
}

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_
