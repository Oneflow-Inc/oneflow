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
#include "oneflow/core/operator/return_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

void ReturnOp::InitFromOpConf() {
  CHECK(op_conf().has_return_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_is_mutable(true);
}

namespace {

Maybe<void> InferBlobDescs(const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp) {
  *BlobDesc4BnInOp("out") = *BlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> ReturnOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  return InferBlobDescs(BlobDesc4BnInOp);
}

Maybe<void> ReturnOp::InferOutBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  return InferBlobDescs(GetBlobDesc4BnInOp);
}

Maybe<void> ReturnOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  const auto& in_sbp_infer_hint = *JUST(SbpInferHint4Ibn("in"));
  CHECK_EQ_OR_RETURN(in_sbp_infer_hint.parallel_desc().parallel_num(),
                     parallel_desc.parallel_num());
  if (in_sbp_infer_hint.sbp_parallel().has_partial_sum_parallel()) {
    SbpSignatureBuilder().Broadcast(input_bns()).Broadcast(output_bns()).Build(sbp_signature);
  } else {
    auto* bn2sbp = sbp_signature->mutable_bn_in_op2sbp_parallel();
    (*bn2sbp)["in"] = in_sbp_infer_hint.sbp_parallel();
    (*bn2sbp)["out"] = in_sbp_infer_hint.sbp_parallel();
  }
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> ReturnOp::GetOpConfWithoutOpNameAndLbn() const {
  return SymbolOf(this->op_conf());
}

REGISTER_OP(OperatorConf::kReturnConf, ReturnOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kReturnConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kReturnConf);

}  // namespace oneflow
