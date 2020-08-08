/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#ifndef ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_
#define ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_

#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/high_order_bool.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

namespace user_op {

hob::BoolFunctorPtr<KernelRegContext> HobTrue();

hob::BoolFunctorPtr<KernelRegContext> HobFalse();

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

template<typename ContextT>
class HobStringContextGetter final {
 public:
  HobStringContextGetter(const DeviceType& device_type) {
    std::string str = ToString(device_type);
    debug_str_ = str;
    context_getter_ = [str](const ContextT&) -> const std::string& { return str; };
  }
  HobStringContextGetter(const char* const_value) {
    std::string str(const_value);
    debug_str_ = str;
    context_getter_ = [str](const ContextT&) -> const std::string& { return str; };
  }
  HobStringContextGetter(const std::string& const_value)
      : debug_str_(const_value),
        context_getter_(
            [const_value](const ContextT&) -> const std::string& { return const_value; }) {}
  HobStringContextGetter(const std::string& debug_str,
                         const std::function<const std::string&(const ContextT&)>& context_getter)
      : debug_str_(debug_str), context_getter_(context_getter) {}

  hob::BoolFunctorPtr<ContextT> operator==(const HobStringContextGetter& other) const {
    std::ostringstream string_stream;
    string_stream << debug_str_ << " == " << other.debug_str_;
    std::function<std::string(const ContextT&)> l_fn = this->context_getter_;
    std::function<std::string(const ContextT&)> r_fn = other.context_getter_;
    std::shared_ptr<const hob::BoolFunctor<ContextT>> krbf_ptr =
        std::make_shared<const hob::HighOrderBoolFunctor<ContextT>>(
            string_stream.str(),
            [l_fn, r_fn](const ContextT& ctx) { return l_fn(ctx) == r_fn(ctx); });
    return krbf_ptr;
  }

 private:
  std::string debug_str_;
  std::function<const std::string&(const ContextT&)> context_getter_;
};

HobStringContextGetter<KernelRegContext> HobDeviceTag();

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_USER_OP_HOB_H_
