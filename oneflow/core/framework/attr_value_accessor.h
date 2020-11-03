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
#ifndef ONEFLOW_CORE_FRAMEWORK_ATTR_VAL_ACCESSOR_H_
#define ONEFLOW_CORE_FRAMEWORK_ATTR_VAL_ACCESSOR_H_

#include "oneflow/core/framework/attr_value.h"

namespace oneflow {

namespace user_op {

template<typename T>
struct AttrValueAccessor final {
  static T Attr(const AttrValue&);
  static void Attr(const T&, AttrValue*);
};

}  // namespace user_op

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_ATTR_VAL_ACCESSOR_H_
