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
#include "oneflow/core/operator/source_tick_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void SourceTickOp::InitFromOpConf() {
  CHECK(op_conf().has_source_tick_conf());
  CHECK(op_conf().ctrl_in_op_name().empty());
  EnrollOutputBn("out", false);
}

LogicalNode* SourceTickOp::NewProperLogicalNode() const { return new SourceTickLogicalNode(); }

Maybe<void> SourceTickOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  BlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

Maybe<void> SourceTickOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  GetBlobDesc4BnInOp("out")->mut_shape() = Shape({1});
  return Maybe<void>::Ok();
}

Maybe<void> SourceTickOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignatureBuilder().Broadcast(output_bns()).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

REGISTER_CPU_OP(OperatorConf::kSourceTickConf, SourceTickOp);
REGISTER_TICK_TOCK_OP(OperatorConf::kSourceTickConf);

}  // namespace oneflow
