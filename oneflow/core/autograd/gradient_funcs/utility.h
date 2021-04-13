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

#ifndef ONEFLOW_CORE_AUTOGRAD_GRADIENT_FUNCS_UTILITY_H_
#define ONEFLOW_CORE_AUTOGRAD_GRADIENT_FUNCS_UTILITY_H_

#include <string>

#include "oneflow/core/framework/attr_value_accessor.h"
#include "oneflow/core/framework/user_op_conf.pb.h"

namespace oneflow {

template<typename T>
inline T GetAttr(const UserOpConf& conf, const std::string& attr_name) {
  const auto& it = conf.attr().find(attr_name);
  CHECK(it != conf.attr().end());
  return user_op::AttrValueAccessor<T>::Attr(it->second);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_AUTOGRAD_GRADIENT_FUNCS_UTILITY_H_
