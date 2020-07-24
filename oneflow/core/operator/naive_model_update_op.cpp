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
#include "oneflow/core/operator/naive_model_update_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

const PbMessage& NaiveModelUpdateOp::GetCustomizedConf() const {
  return op_conf().naive_model_update_conf();
}

const HashSet<std::string> NaiveModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{};
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kNaiveConf, NormalModelUpdtOp, NaiveModelUpdateOp);

REGISTER_OP(OperatorConf::kNaiveModelUpdateConf, NaiveModelUpdateOp);

}  // namespace oneflow
