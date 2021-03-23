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

Maybe<void> ParseParallelDistributionFromConf(const VariableOpConf& conf,
                                              const ParallelDesc& parallel_desc,
                                              ParallelDistribution* parallel_distribution) {
  const bool has_parallel_distribution_conf = (conf.parallel_distribution_size() != 0);
  const int64_t num_axes = parallel_desc.hierarchy()->NumAxes();
  if (has_parallel_distribution_conf) { CHECK_EQ(conf.parallel_distribution_size(), num_axes); }
  FOR_RANGE(int64_t, i, 0, num_axes) {
    if (has_parallel_distribution_conf) {
      SbpParallel sbp_parallel;
      CHECK_OR_RETURN(ParseSbpParallelFromString(conf.parallel_distribution(i), &sbp_parallel));
      CHECK_OR_RETURN(sbp_parallel.has_split_parallel() || sbp_parallel.has_broadcast_parallel());
      *parallel_distribution->add_sbp_parallel() = sbp_parallel;
    } else {
      parallel_distribution->add_sbp_parallel()->mutable_broadcast_parallel();
    }
  }
  return Maybe<void>::Ok();
}

}  // namespace

void VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  if (op_conf().variable_conf().has_tick()) { EnrollInputBn("tick", false); }
  bool is_trainable = op_conf().variable_conf().trainable();
  EnrollOutputBn("out", is_trainable)->set_is_mutable(true);
}

Maybe<void> VariableOp::InferLogicalOutBlobDescs(
    const std::function<BlobDesc*(const std::string&)>& BlobDesc4BnInOp,
    const ParallelDesc& parallel_desc) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  BlobDesc* out_blob_desc = BlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(variable_conf.shape());
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
  ParallelDistribution parallel_distribution;
  JUST(ParseParallelDistributionFromConf(variable_conf, parallel_desc, &parallel_distribution));
  out_blob_desc->mut_shape() = *JUST(GetPhysicalShape(
      Shape(variable_conf.shape()), parallel_distribution, parallel_desc, *parallel_ctx));
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  CHECK_EQ_OR_RETURN(JUST(GetOpParallelDesc())->hierarchy()->NumAxes(), 1);
  SbpSignatureBuilder sbp_sig_builder;
  if (op_conf().variable_conf().parallel_distribution_size() != 0) {
    CHECK_EQ_OR_RETURN(op_conf().variable_conf().parallel_distribution_size(), 1);
    SbpParallel sbp_parallel;
    CHECK_OR_RETURN(ParseSbpParallelFromString(op_conf().variable_conf().parallel_distribution(0),
                                               &sbp_parallel));
    if (sbp_parallel.has_split_parallel()) {
      sbp_sig_builder.Split(output_bns(), sbp_parallel.split_parallel().axis());
    } else {
      sbp_sig_builder.Broadcast(output_bns());
    }
  } else {
    sbp_sig_builder.Broadcast(output_bns());
  }
  sbp_sig_builder.Broadcast(input_bns()).Build(sbp_sig_list->mutable_sbp_signature()->Add());
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::InferSbpSignature(
    SbpSignature* sbp_signature, const SbpSignature& sbp_sig_conf,
    const std::function<int32_t(const SbpSignature&)>& CalcOrderValue4SbpSig,
    std::function<Maybe<const SbpInferHint*>(const std::string&)> SbpInferHint4Ibn,
    const ParallelDesc& parallel_desc) const {
  CHECK_EQ_OR_RETURN(parallel_desc.hierarchy()->NumAxes(), 1);
  SbpSignatureList sbp_sig_list;
  JUST(GetSbpSignatures(&sbp_sig_list));
  *sbp_signature = sbp_sig_list.sbp_signature().Get(0);
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> VariableOp::GetOpConfWithoutOpNameAndLbn() const {
  return SymbolOf(this->op_conf());
}

Maybe<void> VariableOp::InferParallelDistributionSignature(
    ParallelDistributionSignature* parallel_distribution_signature,
    const ParallelDistributionSignature& parallel_distribution_constraints,
    const ParallelDesc& parallel_desc,
    std::function<Maybe<const ParallelDistributionInferHint*>(const std::string&)>
        ParallelDistributionInferHint4Ibn) const {
  const auto& parallel_hierarchy = parallel_desc.hierarchy();
  const VariableOpConf& conf = this->op_conf().variable_conf();
  ParallelDistribution& out_parallel_distribution =
      (*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())["out"];
  JUST(ParseParallelDistributionFromConf(conf, parallel_desc, &out_parallel_distribution));
  if (conf.has_tick()) {
    ParallelDistribution& tick_parallel_distribution =
        (*parallel_distribution_signature->mutable_bn_in_op2parallel_distribution())["tick"];
    for (int64_t i = 0; i < parallel_hierarchy->NumAxes(); ++i) {
      tick_parallel_distribution.mutable_sbp_parallel()->Add()->mutable_broadcast_parallel();
    }
  }
  return Maybe<void>::Ok();
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kVariableConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kVariableConf);

}  // namespace oneflow
