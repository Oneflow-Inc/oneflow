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
#include "oneflow/core/operator/output_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/operator/interface_op_util.h"

namespace oneflow {

void OutputOp::InitFromOpConf() {
  CHECK(op_conf().has_output_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_is_mutable(true);
}

Maybe<void> OutputOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  BlobDesc* out_blob_desc = BlobDesc4BnInOp("out");
  JUST(InterfaceOpUtil::InferLogicalOutBlobDesc(op_conf().output_conf().blob_conf(), out_blob_desc,
                                                parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  if (in_blob_desc->is_dynamic()) {
    *out_blob_desc = *in_blob_desc;
  } else {
    JUST(InterfaceOpUtil::InferOutBlobDesc(op_conf().output_conf().blob_conf(), out_blob_desc,
                                           parallel_ctx, *JUST(GetOpParallelDesc())));
    CHECK_OR_RETURN(*out_blob_desc == *in_blob_desc);
  }
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  JUST(InterfaceOpUtil::GetOutputLikeOpSbpSignature(op_conf().output_conf().blob_conf(),
                                                    input_bns(), output_bns(), sbp_signature));
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::InferParallelDistributionSignature(
    ParallelDistributionSignature* parallel_distribution_signature,
    const ParallelDistributionSignature& parallel_distribution_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
        ParallelDistributionInferHint4Ibn) const {
  const InterfaceBlobConf& blob_conf = op_conf().output_conf().blob_conf();
  ParallelDistribution& in_parallel_distribution =
      (*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())["in"];
  ParallelDistribution& out_parallel_distribution =
      (*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())["out"];
  JUST(InterfaceOpUtil::ParseParallelDistributionFromBlobConf(blob_conf, parallel_desc,
                                                              &in_parallel_distribution));
  JUST(InterfaceOpUtil::ParseParallelDistributionFromBlobConf(blob_conf, parallel_desc,
                                                              &out_parallel_distribution));

  return Maybe<void>::Ok();
}

Symbol<OperatorConf> OutputOp::GetOpConfWithoutOpNameAndLbn() const {
  return SymbolOf(this->op_conf());
}

REGISTER_OP(OperatorConf::kOutputConf, OutputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kOutputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kOutputConf);

}  // namespace oneflow
