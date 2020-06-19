#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_

#include "oneflow/core/common/high_order_bool.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/kernel_registration.h"

namespace oneflow {

namespace user_op {

hob::BoolFunctorPtr<KernelRegContext> HobTrue();

hob::BoolFunctorPtr<KernelRegContext> HobFalse();

hob::HobContextGetter<KernelRegContext, DeviceType> HobDeviceType();

hob::HobContextGetter<KernelRegContext, DataType> HobDataType(const std::string& tensor_name,
                                                              int tensor_idx);

template<typename T>
hob::HobContextGetter<user_op::KernelRegContext, T> HobCtxGetter(
    const std::string& debug_str,
    const std::function<T(const user_op::KernelRegContext&)> hob_func) {
  return hob::HobContextGetter<user_op::KernelRegContext, T>(debug_str, hob_func);
}

template<typename T>
hob::HobContextGetter<user_op::KernelRegContext, T> HobAttr(const std::string& attr_name) {
  return user_op::HobCtxGetter<T>(attr_name, [attr_name](const user_op::KernelRegContext& ctx) {
    return ctx.Attr<T>(attr_name);
  });
}

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_
