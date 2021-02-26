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
#include "oneflow/core/operator/wait_and_send_ids_op.h"
#include "oneflow/core/graph/logical_node.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void WaitAndSendIdsOp::InitFromOpConf() {
  CHECK(op_conf().has_wait_and_send_ids_conf());
  EnrollOutputBn("out", false);
}

LogicalNode* WaitAndSendIdsOp::NewProperLogicalNode() const {
  return new WaitAndSendIdsLogicalNode();
}

namespace {

Maybe<void> InferBlobDescs(const OperatorConf& op_conf,
                           const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  BlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  BlobDesc4BnInOp("out")->set_data_type(op_conf.wait_and_send_ids_conf().data_type());
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> WaitAndSendIdsOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), 1);
  return InferBlobDescs(op_conf(), BlobDesc4BnInOp);
}

Maybe<void> WaitAndSendIdsOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  return InferBlobDescs(op_conf(), GetBlobDesc4BnInOp);
}

Maybe<void> WaitAndSendIdsOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Broadcast(output_bns()).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kWaitAndSendIdsConf, WaitAndSendIdsOp);

}  // namespace oneflow
