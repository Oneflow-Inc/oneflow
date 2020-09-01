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
#include "oneflow/core/operator/rmsprop_model_update_op.h"

namespace oneflow {

void RMSPropModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  const auto& rmsprop_conf = op_conf().rmsprop_model_update_conf().user_conf().rmsprop_conf();
  EnrollInputBn("mean_square", false)->set_is_mutable(true);
  if (rmsprop_conf.centered()) { EnrollInputBn("mean_gradient", false)->set_is_mutable(true); }
}

Maybe<void> RMSPropModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& rmsprop_conf = op_conf().rmsprop_model_update_conf().user_conf().rmsprop_conf();
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ_OR_RETURN(model_blob_desc->data_type(), job_desc().DefaultDataType());
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("mean_square") == *model_blob_desc);
  if (rmsprop_conf.centered()) {
    CHECK_OR_RETURN(*GetBlobDesc4BnInOp("mean_gradient") == *model_blob_desc);
  }
  return Maybe<void>::Ok();
}

const PbMessage& RMSPropModelUpdateOp::GetCustomizedConf() const {
  return op_conf().rmsprop_model_update_conf();
}

const HashSet<std::string> RMSPropModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{};
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kRmspropConf, NormalModelUpdtOp, RMSPropModelUpdateOp);

REGISTER_OP(OperatorConf::kRmspropModelUpdateConf, RMSPropModelUpdateOp);

}  // namespace oneflow
