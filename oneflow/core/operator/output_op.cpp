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
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/operator/output_op.h"
#include "oneflow/core/job/sbp_signature_builder.h"
#include "oneflow/core/job/env_desc.h"

namespace oneflow {

namespace {
Maybe<void> InferOutputOpNdSbpSignature(NdSbpSignature* nd_sbp_signature,
                                        const ParallelDesc& parallel_desc,
                                        const OperatorConf& op_conf) {
  const InterfaceBlobConf& blob_conf = op_conf.output_conf().blob_conf();
  NdSbp& in_nd_sbp = (*nd_sbp_signature->mutable_bn_in_op2nd_sbp())["in"];
  NdSbp& out_nd_sbp = (*nd_sbp_signature->mutable_bn_in_op2nd_sbp())["out"];
  JUST(InterfaceOpUtil::ParseNdSbpFromBlobConf(blob_conf, parallel_desc, &in_nd_sbp));
  JUST(InterfaceOpUtil::ParseNdSbpFromBlobConf(blob_conf, parallel_desc, &out_nd_sbp));
  return Maybe<void>::Ok();
}
}  // anonymous namespace

Maybe<void> OutputOp::InitFromOpConf() {
  CHECK(op_conf().has_output_conf());
  EnrollInputBn("in");
  EnrollOutputBn("out")->set_is_mutable(true);
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  BlobDesc* out_blob_desc = BlobDesc4BnInOp("out");
  *out_blob_desc = *BlobDesc4BnInOp("in");
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const BlobDesc* in_blob_desc = GetBlobDesc4BnInOp("in");
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  // NOTE(chengcheng):
  //   In multi-client, in blob shape maybe changed and NOT equal with output_conf.blob_conf,
  //   and the output op actually is return op (used in single-client) with NO blob conf.
  *out_blob_desc = *in_blob_desc;
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  SbpSignature* sbp = sbp_sig_list->mutable_sbp_signature()->Add();
  CHECK_EQ_OR_RETURN(JUST(GetOpParallelDesc())->hierarchy()->NumAxes(), 1)
      << "Only support 1d sbp now.";
  // Get sbp from BlobConf
  const InterfaceBlobConf& blob_conf = op_conf().output_conf().blob_conf();
  // TODO: make sure blob_conf must set nd_sbp
  CHECK_OR_RETURN(blob_conf.has_nd_sbp());
  const SbpParallel& sbp_parallel = SbpParallel(blob_conf.nd_sbp().sbp_parallel(0));
  if (sbp_parallel.has_broadcast_parallel()) {
    SbpSignatureBuilder().Broadcast("in").Broadcast("out").Build(sbp);
  } else if (sbp_parallel.has_partial_sum_parallel()) {
    SbpSignatureBuilder().PartialSum("in").PartialSum("out").Build(sbp);
  } else if (sbp_parallel.has_split_parallel()) {
    int64_t split_axis = sbp_parallel.split_parallel().axis();
    SbpSignatureBuilder().Split("in", split_axis).Split("out", split_axis).Build(sbp);
  } else {
    UNIMPLEMENTED_THEN_RETURN();
  }
  return Maybe<void>::Ok();
}

Maybe<void> OutputOp::GetNdSbpSignatureList(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    const ParallelDesc& parallel_desc, std::vector<NdSbpSignature>* nd_sbp_sig_list) const {
  NdSbpSignature nd_sbp_signature;
  JUST(InferOutputOpNdSbpSignature(&nd_sbp_signature, parallel_desc, op_conf()));
  nd_sbp_sig_list->emplace_back(nd_sbp_signature);
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

Maybe<void> OutputOp::InferNdSbpSignature(
    NdSbpSignature* nd_sbp_signature, const NdSbpSignature& nd_sbp_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const NdSbpInferHint*>(const std::string&)> NdSbpInferHint4Ibn) const {
  JUST(InferOutputOpNdSbpSignature(nd_sbp_signature, parallel_desc, op_conf()));
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> OutputOp::GetOpConfWithoutOpNameAndLbn() const {
  return SymbolOf(this->op_conf());
}

REGISTER_OP(OperatorConf::kOutputConf, OutputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kOutputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kOutputConf);

}  // namespace oneflow
