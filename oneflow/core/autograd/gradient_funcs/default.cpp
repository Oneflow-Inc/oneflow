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

#include "oneflow/core/framework/op_expr_grad_function.h"

#include "oneflow/core/framework/op_builder.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter/op_interpreter_util.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {
namespace one {

class DefaultOpExprGradFunction : public OpExprGradFunction {
 public:
  Maybe<void> Init(const OpExpr& op) override;

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  Maybe<void> GenerateOpGradConf(const Operator& op, std::vector<OperatorConf>* bw_op_confs);

 private:
  std::shared_ptr<Operator> fw_adapter_op_;

  struct BackwardEntry {
    std::shared_ptr<OpExpr> backward_op;
    std::shared_ptr<Operator> bw_adapter_op;
  };
  std::vector<BackwardEntry> backward_entries_;

  // The input gradient logical blob id for each forward input blob name which
  // needs backward.
  HashMap<std::string, LogicalBlobId> ibn_to_grad_lbi_map_;
  // The indexed input blob names.
  std::vector<std::string> indexed_ibns_;
  // captured inputs
  std::vector<std::string, int64_t> fw_ibn_index_and_saved_index_pairs_;

  // The output gradient logical blob id for each forward output blob name.
  HashMap<std::string, LogicalBlobId> obn_to_grad_lbi_map_;
  // The indexed output blob names.
  std::vector<std::string> indexed_obns_;
  // captured outputs
  std::vector<std::string, int64_t> fw_obn_index_and_saved_index_pairs_;
};

namespace {

Maybe<void> DefaultOpExprGradFunction::GenerateOpGradConf(const Operator& op,
                                                          std::vector<OperatorConf>* bw_op_confs) {
  auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
    const auto& input_bns = op.input_bns();
    const auto& output_bns = op.output_bns();
    if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
      return &ibn_to_grad_lbi_map_[bn];
    } else if (std::find(output_bns.begin(), output_bns.end(), bn) != output_bns.end()) {
      auto it = obn_to_grad_lbi_map_.find(bn);
      if (it == obn_to_grad_lbi_map_.end()) {
        LogicalBlobId lbi;
        lbi.set_op_name(GradientOpName(op.op_name()));
        lbi.set_blob_name(bn);
        it = obn_to_grad_lbi_map_.emplace(bn, lbi).first;
      }
      return &(it->second);
    } else {
      LOG(FATAL) << "Blob name (" << op.op_name() << "/" << bn << ") is missing.";
    }
    return nullptr;
  };
  const auto& dummy_blob_desc = BlobDesc(Shape(), DataType::kInvalidDataType);
  auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
    return dummy_blob_desc;
  };
  JUST(GenerateBackwardOpConfIf(op, bw_op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp));
  return Maybe<void>::Ok();
}

Maybe<void> DefaultOpExprGradFunction::Init(const OpExpr& op) {
  if (op.input_num() == 0) { return Maybe<void>::Ok(); }
  const auto* fw_op_expr = dynamic_cast<const BuiltinOpExpr*>(&op);
  CHECK_NOTNULL_OR_RETURN(fw_op_expr);
  indexed_ibns_ = fw_op_expr->indexed_ibns();
  indexed_obns_ = fw_op_expr->indexed_obns();
  OperatorConf fw_op_conf;
  fw_op_expr->BuildOpConf(&fw_op_conf);

  // Generate backward operator conf for each input. The `LogicalBlobId` for
  // backward output gradient is dummy due to inaccessibility.
  fw_adapter_op_ = ConstructOp(fw_op_conf, DeviceType::kCPU);
  std::vector<OperatorConf> bw_op_confs;
  JUST(GenerateOpGradConf(*fw_adapter_op_, &bw_op_confs));

  CHECK_EQ_OR_RETURN(op.input_num(), ibn_to_grad_lbi_map_.size())
      << "All inputs are considered to require gradients since the `requires_grad` "
         "has been unknown to us here.";
  backward_entries_.resize(bw_op_confs.size());

  for (int i = 0; i < bw_op_confs.size(); ++i) {
    BackwardEntry* entry = &(backward_entries_[i]);
    const auto& op_conf = bw_op_confs.at(i);
    entry->bw_adapter_op = ConstructOp(op_conf, DeviceType::kCPU);
    std::vector<std::string> bw_indexed_ibns;
    {
      const auto& input_bns = entry->bw_adapter_op->input_bns();
      bw_indexed_ibns = {input_bns.begin(), input_bns.end()};
    }
    std::vector<std::string> bw_indexed_obns;
    {
      const auto& output_bns = entry->bw_adapter_op->output_bns();
      bw_indexed_obns = {output_bns.begin(), output_bns.end()};
    }
    // Currently only user op is considered.
    CHECK(op_conf.has_user_conf());
    UserOpConf user_conf(op_conf.user_conf());
    entry->backward_op = std::make_shared<UserOpExpr>(op_conf.name(), std::move(user_conf),
                                                      bw_indexed_ibns, bw_indexed_obns);
  }
  {
    HashSet<LogicalBlobId> fw_input_lbis;
    for (const auto& ibn : fw_adapter_op_->input_bns()) {
      fw_input_lbis.insert(fw_adapter_op_->BnInOp2Lbi(ibn));
    }
    HashSet<LogicalBlobId> fw_output_lbis;
    for (const auto& obn : fw_adapter_op_->output_bns()) {
      fw_output_lbis.insert(fw_adapter_op_->BnInOp2Lbi(obn));
    }
    HashSet<LogicalBlobId> captured_fw_input_lbis;
    HashSet<LogicalBlobId> captured_fw_output_lbis;
    const auto& UpdateCapturedLbis = [&](const LogicalBlobId& lbi) {
      if (fw_input_lbis.count(lbi) > 0) { captured_fw_input_lbis.insert(lbi); }
      if (fw_output_lbis.count(lbi) > 0) { captured_fw_output_lbis.insert(lbi); }
    };
    for (const auto& entry : backward_entries_) {
      const auto& bw_op = *entry.bw_adapter_op;
      for (const auto& ibn : bw_op.input_bns()) { UpdateCapturedLbis(bw_op.BnInOp2Lbi(ibn)); }
      for (const auto& obn : bw_op.output_bns()) { UpdateCapturedLbis(bw_op.BnInOp2Lbi(obn)); }
    }
    int captured_index = 0;
    for (int input_index = 0; input_index < indexed_ibns_.size(); ++input_index) {
      const auto& ibn = indexed_ibns_.at(input_index);
      if (captured_fw_input_lbis.count(fw_adapter_op_.BnInOp2Lbi(ibn)) > 0) {
        fw_ibn_index_and_saved_index_pairs_.push_back(
            std::make_pair(input_index, captured_index++));
      }
    }
    for (int output_index = 0; output_index < indexed_obns_.size(); ++output_index) {
      const auto& obn = indexed_obns_.at(output_index);
      if (captured_fw_output_lbis.count(fw_adapter_op_.BnInOp2Lbi(obn)) > 0) {
        fw_obn_index_and_saved_index_pairs_.push_back(
            std::make_pair(indexed_obns_, captured_index++));
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DefaultOpExprGradFunction::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                                               const TensorTuple& outputs) const {
  for (const auto& pair : fw_ibn_index_and_saved_index_pairs_) {
    CHECK_EQ(ctx->SavedTensors().size(), pair.second);
    ctx->SaveTensorForBackward(inputs.at(pair.first));
  }
  for (const auto& pair : fw_obn_index_and_saved_index_pairs_) {
    CHECK_EQ(ctx->SavedTensors().size(), pair.second);
    ctx->SaveTensorForBackward(outputs.at(pair.first));
  }
  return Maybe<void>::Ok();
}

Maybe<void> DefaultOpExprGradFunction::Apply(const OpExprInterpState* ctx,
                                             const TensorTuple& out_grads,
                                             TensorTuple* in_grads) const {
  HashMap<LogicalBlobId, std::shared_ptr<one::Tensor>> lbi2tensor;
  const auto& saved_tensors = ctx->SavedTensors();
  // Fills lbi2tensor by captured forward inputs
  for (const auto& pair : fw_ibn_index_and_saved_index_pairs_) {
    const auto& lbi = fw_adapter_op->BnInOp2Lbi(indexed_ibns_.at(pair.first));
    CHECK(lbi2tensor.emplace(lbi, saved_tensors.at(pair.second)).second);
  }
  // Fills lbi2tensor by captured forward outputs
  for (const auto& pair : fw_obn_index_and_saved_index_pairs_) {
    const auto& lbi = fw_adapter_op->BnInOp2Lbi(indexed_obns_.at(pair.first));
    CHECK(lbi2tensor.emplace(lbi, saved_tensors.at(pair.second)).second);
  }
  // Fills lbi2tensor by captured backward out_grads
  for (int i = 0; i < indexed_obns_.size(); ++i) {
    const auto& lbi = obn_to_grad_lbi_map_.at(indexed_obns_.at(i));
    CHECK(lbi2tensor.emplace(lbi, out_grads.at(i)).second);
  }
  for (const auto& entry : backward_entries_) {
    const auto& bw_adapter_op = entry->bw_adapter_op;
    const auto& bw_op_expr = entry->backward_op;
    const auto& indexed_ibns = bw_op_expr->indexed_ibns();
    // Init inputs
    TensorTuple inputs(indexed_ibns.size());
    for (int i = 0; i < indexed_ibns.size(); ++i) {
      const auto& ibn = indexed_ibns.at(i);
      const auto& lbi = bw_adapter_op.BnInOp2Lbi(ibn);
      inputs.at(i) = lbi2tensor.at(lbi);
    }
    const auto& indexed_obns = bw_op_expr->indexed_obns();
    TensorTuple outputs(indexed_obns.size());
    const auto& interpreter = JUST(OpInterpUtil::GetInterpreter());
    JUST(interpreter->Apply(*bw_op_expr, inputs, &outputs));
    // Update lbi2tensor by results
    for (int i = 0; i < indexed_obns.size(); ++i) {
      const auto& obn = indexed_obns.at(i);
      const auto& lbi = bw_adapter_op.BnInOp2Lbi(obn);
      CHECK(lbi2tensor.emplace(lbi, outputs.at(i)).second);
    }
  }
  // Updates the result in_grads
  in_grads->resize(indexed_ibns_.size());
  for (int i = 0; i < indexed_ibns_.size(); ++i) {
    const auto& lbi = ibn_to_grad_lbi_map_.at(indexed_ibns_.at(i));
    in_grads->at(i) = lbi2tensor.at(lbi);
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("default", DefaultOpExprGradFunction);

}  // namespace one
}  // namespace oneflow
