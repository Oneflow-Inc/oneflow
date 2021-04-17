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
#ifndef ONEFLOW_CORE_FRAMEWORK_OP_TYPE_TRAIT_H_
#define ONEFLOW_CORE_FRAMEWORK_OP_TYPE_TRAIT_H_

#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

template<OperatorConf::OpTypeCase op_type_case>
struct OpTypeTrait {};

#define INSTANCE_OP_TYPE_TRAIT(op_type_case, _proto_type, _op_type_name) \
  template<>                                                             \
  struct OpTypeTrait<op_type_case> {                                     \
    using proto_type = _proto_type;                                      \
    static const std::string op_type_name() { return _op_type_name; }    \
  };

INSTANCE_OP_TYPE_TRAIT(OperatorConf::kUserConf, UserOpConf, "user");
INSTANCE_OP_TYPE_TRAIT(OperatorConf::kVariableConf, VariableOpConf, "variable");
INSTANCE_OP_TYPE_TRAIT(OperatorConf::kCastToMirroredConf, CastToMirroredOpConf, "cast_to_mirrored");
INSTANCE_OP_TYPE_TRAIT(OperatorConf::kCastFromMirroredConf, CastFromMirroredOpConf,
                       "cast_from_mirrored");
INSTANCE_OP_TYPE_TRAIT(OperatorConf::kDistributeSplitConf, DistributeSplitOpConf,
                       "distribute_split");
INSTANCE_OP_TYPE_TRAIT(OperatorConf::kDistributeCloneConf, DistributeCloneOpConf,
                       "distribute_clone");
INSTANCE_OP_TYPE_TRAIT(OperatorConf::kDistributeConcatConf, DistributeConcatOpConf,
                       "distribute_concat");
INSTANCE_OP_TYPE_TRAIT(OperatorConf::kDistributeAddConf, DistributeAddOpConf, "distribute_add");

#undef INSTANCE_OP_TYPE_TRAIT

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_OP_TYPE_TRAIT_H_
