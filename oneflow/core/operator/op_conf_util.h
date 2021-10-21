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
#ifndef ONEFLOW_CORE_OPERATOR_OP_CONF_UTIL_H_
#define ONEFLOW_CORE_OPERATOR_OP_CONF_UTIL_H_

#include "oneflow/core/operator/op_conf.pb.h"

namespace std {

template<>
struct hash<::oneflow::OperatorConf::OpTypeCase> {
  std::size_t operator()(const ::oneflow::OperatorConf::OpTypeCase& op_type) const {
    return std::hash<int>()(static_cast<size_t>(op_type));
  }
};

}  // namespace std

#endif  // ONEFLOW_CORE_OPERATOR_OP_CONF_UTIL_H_
