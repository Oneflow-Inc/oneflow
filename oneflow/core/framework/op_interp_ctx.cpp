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
#include "oneflow/core/framework/op_interp_ctx.h"
#include "oneflow/core/framework/attr_value.h"

namespace oneflow {

Maybe<AttrVal> OpInterpCtx::GetAttr(const std::string& attr_name) const {
  return op_->GetAttr(attr_name);
}

template<typename T>
Maybe<const T&> OpInterpCtx::GetAttr(const std::string& attr_name) const {
  const auto& attr_val = JUST(this->GetAttr(attr_name));
  if (const auto* ptr = dynamic_cast<const user_op::TypedAttrValRef<T>*>(attr_val.get())) {
    return ptr->val();
  }
  return Error::RuntimeError() << "Invalid type for attribute " << attr_name;
}

OpAttrs OpInterpCtx::GetAttrs() const { return OpAttrs(this); }

template<typename T>
Maybe<void> OpInterpCtx::SetAttr(const std::string& attr_name, const T& attr_val) {
  *const_cast<T*>(&JUST(this->GetAttr<T>(attr_name))) = attr_val;
  return Maybe<void>::Ok();
}

#define INSTANCE_ATTR_GETTER_AND_SETTER(field, T, attr_type)                         \
  template Maybe<const T&> OpInterpCtx::GetAttr(const std::string& attr_name) const; \
  template Maybe<void> OpInterpCtx::SetAttr(const std::string& attr_name, const T& attr_val);

OF_PP_FOR_EACH_TUPLE(INSTANCE_ATTR_GETTER_AND_SETTER, ATTR_SEQ)
#undef INSTANCE_ATTR_GETTER_AND_SETTER

Maybe<void> OpInterpCtx::SetAttr(const std::string& attr_name, const AttrVal& attr_val) {
#define MAKE_ENTRY(field, cpp_type, attr_type)                                               \
  if (const auto* ptr = dynamic_cast<const user_op::TypedAttrValIf<cpp_type>*>(&attr_val)) { \
    return this->SetAttr<cpp_type>(attr_name, ptr->val());                                   \
  }

  OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, ATTR_SEQ);
#undef MAKE_ENTRY
  return Error::RuntimeError() << "Invalid type for attribute " << attr_name;
}

bool OpInterpCtx::HasAttr(const std::string& attr_name) const {
  return AttrNames().count(attr_name) > 0;
}

const HashSet<std::string>& OpInterpCtx::AttrNames() const { return op_->AttrNames(); }

}  // namespace oneflow
