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
#include "oneflow/core/operator/variable_op.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/job/sbp_signature_builder.h"

namespace oneflow {

namespace {

Maybe<void> ParseNdSbpFromConf(const VariableOpConf& conf, const ParallelDesc& parallel_desc,
                               NdSbp* nd_sbp) {
  const bool has_nd_sbp_conf = (conf.nd_sbp_size() != 0);
  const int64_t num_axes = parallel_desc.hierarchy()->NumAxes();
  if (has_nd_sbp_conf) { CHECK_EQ(conf.nd_sbp_size(), num_axes); }
  nd_sbp->clear_sbp_parallel();
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (has_nd_sbp_conf) {
      CHECK_OR_RETURN(ParseSbpParallelFromString(conf.nd_sbp(i), nd_sbp->add_sbp_parallel()));
    } else {
      nd_sbp->add_sbp_parallel()->mutable_broadcast_parallel();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  if (op_conf().variable_conf().has_tick()) { EnrollInputBn("tick", false); }
  bool is_trainable = op_conf().variable_conf().trainable();
  EnrollOutputBn("out", is_trainable)->set_is_mutable(true);
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  BlobDesc* out_blob_desc = BlobDesc4BnInOp("out");
  out_blob_desc->set_shape(Shape(variable_conf.shape()));
  CHECK_OR_RETURN(variable_conf.has_data_type());
  out_blob_desc->set_data_type(variable_conf.data_type());
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::InferOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  const ParallelDesc& parallel_desc = *JUST(GetOpParallelDesc());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  CHECK_OR_RETURN(variable_conf.has_data_type());
  out_blob_desc->set_data_type(variable_conf.data_type());
  NdSbp nd_sbp;
  JUST(ParseNdSbpFromConf(variable_conf, parallel_desc, &nd_sbp));
  out_blob_desc->set_shape(
      *JUST(GetPhysicalShape(Shape(variable_conf.shape()), nd_sbp, parallel_desc, *parallel_ctx)));
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  int64_t num_axes = op_conf().variable_conf().shape().dim_size();
  for (int i = 0; i < num_axes; ++i) {
    SbpSignatureBuilder()
        .Broadcast(input_bns())
        .Split(output_bns(), i)
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.hierarchy()->NumAxes(), 1);
  SbpSignatureBuilder sbp_sig_builder;
  if (op_conf().variable_conf().nd_sbp_size() != 0) {
    CHECK_EQ_OR_RETURN(op_conf().variable_conf().nd_sbp_size(), 1);
    SbpParallel sbp_parallel;
    CHECK_OR_RETURN(ParseSbpParallelFromString(op_conf().variable_conf().nd_sbp(0), &sbp_parallel));
    if (sbp_parallel.has_split_parallel()) {
      sbp_sig_builder.Split(output_bns(), sbp_parallel.split_parallel().axis());
    } else {
      sbp_sig_builder.Broadcast(output_bns());
    }
  } else {
    sbp_sig_builder.Broadcast(output_bns());
  }
  sbp_sig_builder.Broadcast(input_bns()).Build(sbp_signature);
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> VariableOp::GetOpConfWithoutOpNameAndLbn() const {
  return SymbolOf(this->op_conf());
}

Maybe<void> VariableOp::InferNdSbpSignature(
    NdSbpSignature* nd_sbp_signature, const NdSbpSignature& nd_sbp_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const NdSbpInferHint*>(const std::string&)> NdSbpInferHint4Ibn) const {
  const auto& parallel_hierarchy = parallel_desc.hierarchy();
  const VariableOpConf& conf = this->op_conf().variable_conf();
  NdSbp& out_nd_sbp = (*nd_sbp_signature->mutable_bn_in_op2nd_sbp())["out"];
  JUST(ParseNdSbpFromConf(conf, parallel_desc, &out_nd_sbp));
  if (conf.has_tick()) {
    NdSbp& tick_nd_sbp = (*nd_sbp_signature->mutable_bn_in_op2nd_sbp())["tick"];
    for (int64_t i = 0; i < parallel_hierarchy->NumAxes(); ++i) {
      tick_nd_sbp.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
    }
  }
  return Maybe<void>::Ok();
}

Operator::DumpNdSbpSignatureForOpConfFn VariableOp::GetDumpNdSbpSignatureForOpConfFn() const {
  return [](const NdSbpSignature& nd_sbp_sig, OperatorConf* op_conf) -> Maybe<void> {
    CHECK_OR_RETURN(op_conf->has_variable_conf()) << "VariableOp don't set variable op_conf";
    op_conf->mutable_variable_conf()->clear_nd_sbp();
    const auto& nd_sbp = nd_sbp_sig.bn_in_op2nd_sbp().at("out");

    for (const auto& sbp_parallel : nd_sbp.sbp_parallel()) {
      op_conf->mutable_variable_conf()->mutable_nd_sbp()->Add(SbpParallelToString(sbp_parallel));
    }
    return Maybe<void>::Ok();
  };
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kVariableConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kVariableConf);

}  // namespace oneflow
