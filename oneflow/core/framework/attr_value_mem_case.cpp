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
#include "oneflow/core/framework/attr_value_mem_case.h"

namespace oneflow {

template<typename T>
const T& AttrValueCast(const AttrValue& attr_val) {
  const auto* typed_attr = dynamic_cast<const TypeAttrVal<T>*>(&attr_val);
  return CHECK_NOTNULL(typed_attr)->val();
}

template<typename T>
T& AttrValueCastNotConst(AttrValue& attr_val) {
  auto* typed_attr = dynamic_cast<const TypeAttrVal<T>*>(&attr_val);
  return CHECK_NOTNULL(typed_attr)->val();
}


}// namespace oneflow