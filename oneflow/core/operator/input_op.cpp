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
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/operator/input_op.h"
#include "oneflow/core/operator/interface_op_util.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {
Maybe<void> InferInputOpNdSbpSignature(NdSbpSignature* nd_sbp_signature,
                                       const ParallelDesc& parallel_desc,
                                       const OperatorConf& op_conf) {
  const auto& parallel_hierarchy = parallel_desc.hierarchy();
  const InterfaceBlobConf& blob_conf = op_conf.input_conf().blob_conf();
  if (op_conf.input_conf().has_tick()) {
    NdSbp& tick_nd_sbp = (*nd_sbp_signature->mutable_bn_in_op2nd_sbp())["tick"];
    tick_nd_sbp.clear_sbp_parallel();
    FOR_RANGE(int64_t, i, 0, parallel_hierarchy->NumAxes()) {
      tick_nd_sbp.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
    }
  }
  NdSbp& out_nd_sbp = (*nd_sbp_signature->mutable_bn_in_op2nd_sbp())["out"];
  JUST(InterfaceOpUtil::ParseNdSbpFromBlobConf(blob_conf, parallel_desc, &out_nd_sbp));
  return Maybe<void>::Ok();
}
}  // namespace

Maybe<void> InputOp::InitFromOpConf() {
  CHECK(op_conf().has_input_conf());
  if (op_conf().input_conf().has_tick()) { EnrollInputBn("tick", false); }
  OutputBlobModifier* modifier = EnrollOutputBn("out", false);
  modifier->set_is_mutable(true);
  modifier->set_header_infered_before_compute(false);
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  BlobDesc* out_blob_desc = BlobDesc4BnInOp("out");
  JUST(InterfaceOpUtil::InferLogicalOutBlobDesc(op_conf().input_conf().blob_conf(), out_blob_desc,
                                                parallel_desc));
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  JUST(InterfaceOpUtil::InferOutBlobDesc(op_conf().input_conf().blob_conf(), out_blob_desc,
                                         parallel_ctx, *JUST(GetOpParallelDesc())));
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  JUST(InterfaceOpUtil::GetInputLikeOpSbpSignature(op_conf().input_conf().blob_conf(), input_bns(),
                                                   output_bns(), sbp_signature));
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  JUST(InterfaceOpUtil::GetInputLikeOpSbpSignature(op_conf().input_conf().blob_conf(), input_bns(),
                                                   output_bns(),
                                                   sbp_sig_list->mutable_sbp_signature()->Add()));
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::GetNdSbpSignatureList(
    const std::function<Maybe<const BlobDesc&>(const std::string&)>& LogicalBlobDesc4Ibn,
    const ParallelDesc& parallel_desc, std::vector<NdSbpSignature>* nd_sbp_sig_list) const {
  NdSbpSignature nd_sbp_signature;
  JUST(InferInputOpNdSbpSignature(&nd_sbp_signature, parallel_desc, op_conf()));
  nd_sbp_sig_list->emplace_back(nd_sbp_signature);
  return Maybe<void>::Ok();
}

Maybe<void> InputOp::InferNdSbpSignature(
    NdSbpSignature* nd_sbp_signature, const NdSbpSignature& nd_sbp_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const NdSbpInferHint*>(const std::string&)> NdSbpInferHint4Ibn) const {
  JUST(InferInputOpNdSbpSignature(nd_sbp_signature, parallel_desc, op_conf()));
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> InputOp::GetOpConfWithoutOpNameAndLbn() const {
  return SymbolOf(this->op_conf());
}

REGISTER_OP(OperatorConf::kInputConf, InputOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kInputConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kInputConf);

}  // namespace oneflow
