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
#include "oneflow/core/framework/attr_value.h"

namespace oneflow {

template<typename T>
const T& AttrValueCast(const user_op::AttrVal& attr_val) {
  const auto* typed_attr = dynamic_cast<const user_op::TypedAttrValIf<T>*>(&attr_val);
  return CHECK_NOTNULL(typed_attr)->val();
}

template<typename T>
std::shared_ptr<user_op::AttrVal> CastAttrValue(const T& attr_val) {
  return std::make_shared<user_op::TypedAttrVal<T>>(attr_val);
}

template<typename T>
std::shared_ptr<user_op::AttrVal> CastAttrValue(const T* attr_val) {
  return std::make_shared<user_op::TypedAttrValRef<T>>(attr_val);
}

template<typename T>
size_t HashTypedAttrVal(const T& val) {
  return std::hash<T>()(val);
}

#define INITIALIZE_ATTR_VALUE_CAST(field, T, attr_type)                        \
  template const T& AttrValueCast(const user_op::AttrVal& attr_val);           \
  template std::shared_ptr<user_op::AttrVal> CastAttrValue(const T& attr_val); \
  template std::shared_ptr<user_op::AttrVal> CastAttrValue(const T* attr_val); \
  template size_t HashTypedAttrVal(const T& attr_val);

OF_PP_FOR_EACH_TUPLE(INITIALIZE_ATTR_VALUE_CAST, ATTR_SEQ)

}  // namespace oneflow
