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

Maybe<OptInt64> GetSplitAxis(const VariableOpConf& variable_conf) {
  auto opt_split_axis = std::make_shared<OptInt64>(variable_conf.split_axis());
  if (opt_split_axis->has_value()) {
    size_t num_axes = variable_conf.shape().dim_size();
    if (opt_split_axis->value() < 0) {
      opt_split_axis->set_value(opt_split_axis->value() + num_axes);
    }
    CHECK_GE_OR_RETURN(opt_split_axis->value(), 0);
    CHECK_LT_OR_RETURN(opt_split_axis->value(), num_axes);
  }
  return opt_split_axis;
}

}  // namespace

void VariableOp::InitFromOpConf() {
  CHECK(op_conf().has_variable_conf());
  if (op_conf().variable_conf().has_tick()) { EnrollInputBn("tick", false); }
  bool is_trainable = job_desc().IsTrain() && op_conf().trainable();
  EnrollOutputBn("out", is_trainable)->set_is_mutable(true);
}

Maybe<void> VariableOp::InferBlobDescs(
    std::function<BlobDesc*(const std::string&)> GetBlobDesc4BnInOp,
    const ParallelContext* parallel_ctx, const SbpSignature* sbp_signature) const {
  const VariableOpConf& variable_conf = op_conf().variable_conf();
  CHECK_OR_RETURN(job_desc().job_conf().has_default_initializer_conf()
                  || job_desc().job_conf().has_default_initialize_with_snapshot_path()
                  || variable_conf.has_initializer()
                  || variable_conf.has_initialize_with_snapshot());
  BlobDesc* out_blob_desc = GetBlobDesc4BnInOp("out");
  out_blob_desc->mut_shape() = Shape(variable_conf.shape());
  out_blob_desc->set_data_type(variable_conf.has_data_type() ? variable_conf.data_type()
                                                             : job_desc().DefaultDataType());
  const SplitParallel opt_split_axis =
      sbp_signature->bn_in_op2sbp_parallel().at("out").split_parallel();
  if (opt_split_axis.has_axis()) {
    int32_t model_split_axis = opt_split_axis.axis();
    int64_t split_dim_num = out_blob_desc->shape().At(model_split_axis);
    BalancedSplitter bs(split_dim_num, parallel_ctx->parallel_num());
    out_blob_desc->mut_shape().Set(model_split_axis, bs.At(parallel_ctx->parallel_id()).size());
  }
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::UpdateOpconf() {
  const SbpSignature* sbp_sig_conf = JUST(sbp_signature());
  OptInt64* split_axis = mut_op_conf()->mutable_variable_conf()->mutable_split_axis();
  if (sbp_sig_conf->bn_in_op2sbp_parallel().at("out").has_split_parallel())
    split_axis->set_value(sbp_sig_conf->bn_in_op2sbp_parallel().at("out").split_parallel().axis());
  else
    split_axis->clear_value();
  return Maybe<void>::Ok();
}

Maybe<void> VariableOp::InferBatchAxis(
    std::function<OptInt64*(const std::string&)> BatchAxis4BnInOp) const {
  BatchAxis4BnInOp("out")->clear_value();
  return Maybe<void>::Ok();
}

Maybe<double> VariableOp::GetComputeComplexity(
    SbpSignature* sbp_signature,
    std::function<const BlobDesc&(const std::string& bn)> logical_blob_desc4bn,
    const ParallelDesc& parallel_desc) const {
  double CostRatio;
  std::ifstream ifs("/root/work/codes/OneFlow-Benchmark/Classification/cnns/VarCostRatioFile.txt");
  if (ifs.is_open()) {
    ifs >> CostRatio;
  } else
    CostRatio = 1;
  std::cout << "variable Cost Ratio: " << CostRatio << std::endl;
  return CostRatio
         * JUST(Operator::GetComputeComplexity(sbp_signature, logical_blob_desc4bn, parallel_desc));
}

Maybe<void> VariableOp::GetSbpSignatures(SbpSignatureList* sbp_sig_list) const {
  // TODO: re-code this
  // NOTE: It can build all split axis, and delete impossible case in `GetComputeComplexity`
  // build all avaible sbp signature
  for (int32_t i = 0; i < 5; i++) {
    SbpSignatureBuilder sbp_sig_builder;
    sbp_sig_builder.Split(output_bns(), i)
        .Broadcast(input_bns())
        .Build(sbp_sig_list->mutable_sbp_signature()->Add());
  }
  return Maybe<void>::Ok();
}

Symbol<OperatorConf> VariableOp::GetOpConfWithoutOpNameAndLbn() const {
  return SymbolOf(this->op_conf());
}

REGISTER_OP(OperatorConf::kVariableConf, VariableOp);
REGISTER_OP_SAME_OUTPUT_BLOB_REGST_NUM(OperatorConf::kVariableConf, 1);
REGISTER_INTERFACE_OP(OperatorConf::kVariableConf);

}  // namespace oneflow
