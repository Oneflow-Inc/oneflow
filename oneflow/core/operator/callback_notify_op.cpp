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
#include "oneflow/core/operator/callback_notify_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void CallbackNotifyOp::InitFromOpConf() {
  CHECK(op_conf().has_callback_notify_conf());
  EnrollInputBn("in", false);
}

LogicalNode* CallbackNotifyOp::NewProperLogicalNode() const {
  return new CallbackNotifyLogicalNode();
}

Maybe<void> CallbackNotifyOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  CHECK_OR_RETURN(GetBlobDesc4BnInOp("in")->shape() == Shape({1}));
  CHECK_OR_RETURN(IsIntegralDataType(GetBlobDesc4BnInOp("in")->data_type()));
  return Maybe<void>::Ok();
}

Maybe<void> CallbackNotifyOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  return Maybe<void>::Ok();
}

Maybe<void> CallbackNotifyOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Split(input_bns(), 0).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kCallbackNotifyConf, CallbackNotifyOp);

}  // namespace oneflow
