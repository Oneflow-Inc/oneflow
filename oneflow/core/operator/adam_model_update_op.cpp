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
#include "oneflow/core/operator/adam_model_update_op.h"

namespace oneflow {

void AdamModelUpdateOp::MdUpdtVirtualInitFromOpConf() {
  const auto& adam_conf = op_conf().adam_model_update_conf().user_conf().adam_conf();
  CHECK_GE(adam_conf.beta1(), 0);
  CHECK_LT(adam_conf.beta1(), 1);
  CHECK_GE(adam_conf.beta2(), 0);
  CHECK_LT(adam_conf.beta2(), 1);

  EnrollInputBn("m", false)->set_is_mutable(true);
  EnrollInputBn("v", false)->set_is_mutable(true);
  if (adam_conf.do_bias_correction()) {
    EnrollInputBn("beta1_t", false)->set_is_mutable(true);
    EnrollInputBn("beta2_t", false)->set_is_mutable(true);
  }
}

Maybe<void> AdamModelUpdateOp::MdUpdtVirtualInferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const auto& adam_conf = op_conf().adam_model_update_conf().user_conf().adam_conf();
  const BlobDesc* model_blob_desc = GetBlobDesc4BnInOp("model");
  CHECK_EQ_OR_RETURN(model_blob_desc->data_type(), job_desc().DefaultDataType());
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("m") == *model_blob_desc);
  CHECK_OR_RETURN(*GetBlobDesc4BnInOp("v") == *model_blob_desc);

  if (adam_conf.do_bias_correction()) {
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("beta1_t")->shape(), Shape({1}));
    CHECK_EQ_OR_RETURN(GetBlobDesc4BnInOp("beta2_t")->shape(), Shape({1}));
  }
  return Maybe<void>::Ok();
}

const HashSet<std::string> AdamModelUpdateOp::AlwaysBroadcastParallelBns() const {
  return HashSet<std::string>{"beta1_t", "beta2_t"};
}

const PbMessage& AdamModelUpdateOp::GetCustomizedConf() const {
  return op_conf().adam_model_update_conf();
}

REGISTER_CLASS(NormalModelUpdateOpUserConf::kAdamConf, NormalModelUpdtOp, AdamModelUpdateOp);

REGISTER_OP(OperatorConf::kAdamModelUpdateConf, AdamModelUpdateOp);

}  // namespace oneflow
