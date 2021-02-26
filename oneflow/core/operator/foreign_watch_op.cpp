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
#include "oneflow/core/operator/foreign_watch_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

void ForeignWatchOp::InitFromOpConf() {
  CHECK(op_conf().has_foreign_watch_conf());
  EnrollInputBn("in");
}

Maybe<void> ForeignWatchOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), 1);
  return Maybe<void>::Ok();
}

Maybe<void> ForeignWatchOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  CHECK_EQ_OR_RETURN(parallel_ctx->parallel_num(), 1);
  return Maybe<void>::Ok();
}

Maybe<void> ForeignWatchOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.parallel_num(), 1);
  (*sbp_signature->mutable_bn_in_op2sbp_parallel())["in"].mutable_split_parallel()->set_axis(0);
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kForeignWatchConf, ForeignWatchOp);

}  // namespace oneflow
