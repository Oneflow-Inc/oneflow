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

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs, const TensorTuple& outputs,
                      const AttrValueMap& attrs) const override;

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  Maybe<void> GenerateOpGradConf(const Operator& op, std::vector<OperatorConf>* bw_op_confs,
                                 HashMap<std::string, LogicalBlobId>* ibn2grad_lbi,
                                 HashMap<std::string, LogicalBlobId>* obn2grad_lbi);

  std::vector<int64_t> fw_input_index2lbi_symbol_;
  std::vector<int64_t> fw_input_index2grad_lbi_symbol_;
  std::vector<int64_t> fw_output_index2lbi_symbol_;
  std::vector<int64_t> fw_output_index2grad_lbi_symbol_;

  struct BackwardEntry {
    std::shared_ptr<OpExpr> bw_op_expr;
    std::vector<int64_t> bw_input_index2lbi_symbol;
    std::vector<int64_t> bw_output_index2lbi_symbol;
  };
  std::vector<BackwardEntry> backward_entries_;

  // captured inputs
  std::vector<std::pair<int64_t, int64_t>> fw_ibn_index_and_saved_index_pairs_;

  // captured outputs
  std::vector<std::pair<int64_t, int64_t>> fw_obn_index_and_saved_index_pairs_;

  // lbi symbol table
  std::vector<LogicalBlobId> lbi_symbol2lbi_;
};

Maybe<void> DefaultOpExprGradFunction::GenerateOpGradConf(
    const Operator& op, std::vector<OperatorConf>* bw_op_confs,
    HashMap<std::string, LogicalBlobId>* ibn2grad_lbi,
    HashMap<std::string, LogicalBlobId>* obn2grad_lbi) {
  auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
    const auto& input_bns = op.input_bns();
    const auto& output_bns = op.output_bns();
    if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
      return &(*ibn2grad_lbi)[bn];
    } else if (std::find(output_bns.begin(), output_bns.end(), bn) != output_bns.end()) {
      auto it = obn2grad_lbi->find(bn);
      if (it == obn2grad_lbi->end()) {
        LogicalBlobId lbi;
        lbi.set_op_name(GradientOpName(op.op_name()));
        lbi.set_blob_name(bn);
        it = obn2grad_lbi->emplace(bn, lbi).first;
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
  OperatorConf fw_op_conf;
  fw_op_expr->BuildOpConf(&fw_op_conf, /*attrs=*/{});

  // Generate backward operator conf for each input. The `LogicalBlobId` for
  // backward output gradient is dummy due to inaccessibility.
  std::shared_ptr<Operator> fw_adapter_op = ConstructOp(fw_op_conf, DeviceType::kCPU);
  std::vector<OperatorConf> bw_op_confs;

  // The input gradient logical blob id for each forward input blob name
  HashMap<std::string, LogicalBlobId> ibn2grad_lbi;
  // The output gradient logical blob id for each forward output blob name.
  HashMap<std::string, LogicalBlobId> obn2grad_lbi;
  JUST(GenerateOpGradConf(*fw_adapter_op, &bw_op_confs, &ibn2grad_lbi, &obn2grad_lbi));

  CHECK_EQ_OR_RETURN(op.input_num(), ibn2grad_lbi.size())
      << "All inputs are considered to require gradients since the `requires_grad` "
         "has been unknown to us here.";

  std::vector<std::shared_ptr<Operator>> bw_adapter_ops(bw_op_confs.size());
  for (int i = 0; i < bw_op_confs.size(); ++i) {
    const auto& op_conf = bw_op_confs.at(i);
    bw_adapter_ops.at(i) = ConstructOp(op_conf, DeviceType::kCPU);
  }

  // Updates lbi symbol table
  HashMap<LogicalBlobId, int64_t> lbi2lbi_symbol;
  {
    HashSet<LogicalBlobId> lbis;
    for (const auto& ibn : fw_adapter_op->input_bns()) {
      CHECK(lbis.emplace(fw_adapter_op->BnInOp2Lbi(ibn)).second);
    }
    for (const auto& obn : fw_adapter_op->output_bns()) {
      CHECK(lbis.emplace(fw_adapter_op->BnInOp2Lbi(obn)).second);
    }
    for (const auto& bw_adapter_op : bw_adapter_ops) {
      for (const auto& ibn : bw_adapter_op->input_bns()) {
        lbis.insert(bw_adapter_op->BnInOp2Lbi(ibn));
      }
      for (const auto& obn : bw_adapter_op->output_bns()) {
        lbis.insert(bw_adapter_op->BnInOp2Lbi(obn));
      }
    }
    for (const auto& pair : obn2grad_lbi) { lbis.insert(pair.second); }
    lbi_symbol2lbi_ = {lbis.begin(), lbis.end()};
    for (int i = 0; i < lbi_symbol2lbi_.size(); ++i) {
      CHECK(lbi2lbi_symbol.emplace(lbi_symbol2lbi_.at(i), i).second);
    }
  }

  // Initiates backward_entries_
  backward_entries_.resize(bw_adapter_ops.size());
  for (int i = 0; i < bw_adapter_ops.size(); ++i) {
    BackwardEntry* entry = &backward_entries_[i];
    const auto& bw_adapter_op = bw_adapter_ops.at(i);
    const auto& op_conf = bw_adapter_op->op_conf();
    std::vector<std::string> bw_indexed_ibns;
    {
      const auto& input_bns = bw_adapter_op->input_bns();
      bw_indexed_ibns = {input_bns.begin(), input_bns.end()};
    }
    std::vector<std::string> bw_indexed_obns;
    {
      const auto& output_bns = bw_adapter_op->output_bns();
      bw_indexed_obns = {output_bns.begin(), output_bns.end()};
    }
    // Currently only user op is considered.
    CHECK(op_conf.has_user_conf());
    UserOpConf user_conf(op_conf.user_conf());
    // Sets backward op_expr
    entry->bw_op_expr = std::make_shared<UserOpExpr>(op_conf.name(), std::move(user_conf),
                                                     bw_indexed_ibns, bw_indexed_obns);
    // Sets bw_input_index2lbi_symbol
    entry->bw_input_index2lbi_symbol.resize(bw_indexed_ibns.size());
    for (int i = 0; i < bw_indexed_ibns.size(); ++i) {
      const auto& lbi = bw_adapter_op->BnInOp2Lbi(bw_indexed_ibns.at(i));
      entry->bw_input_index2lbi_symbol.at(i) = lbi2lbi_symbol.at(lbi);
    }
    // Sets bw_output_index2lbi_symbol
    entry->bw_output_index2lbi_symbol.resize(bw_indexed_obns.size());
    for (int i = 0; i < bw_indexed_obns.size(); ++i) {
      const auto& lbi = bw_adapter_op->BnInOp2Lbi(bw_indexed_obns.at(i));
      entry->bw_output_index2lbi_symbol.at(i) = lbi2lbi_symbol.at(lbi);
    }
  }
  // Updates fw_xxx_lbi_symbol_
  {
    fw_input_index2lbi_symbol_.resize(fw_op_expr->indexed_ibns().size());
    for (int i = 0; i < fw_op_expr->indexed_ibns().size(); ++i) {
      const auto& ibn = fw_op_expr->indexed_ibns().at(i);
      const auto& lbi = fw_adapter_op->BnInOp2Lbi(ibn);
      fw_input_index2lbi_symbol_.at(i) = lbi2lbi_symbol.at(lbi);
    }
    fw_input_index2grad_lbi_symbol_.resize(fw_op_expr->indexed_ibns().size());
    for (int i = 0; i < fw_op_expr->indexed_ibns().size(); ++i) {
      const auto& ibn = fw_op_expr->indexed_ibns().at(i);
      const auto& grad_lbi = ibn2grad_lbi.at(ibn);
      CHECK(grad_lbi.has_op_name());
      CHECK(grad_lbi.has_blob_name());
      fw_input_index2grad_lbi_symbol_.at(i) = lbi2lbi_symbol.at(grad_lbi);
    }
    fw_output_index2lbi_symbol_.resize(fw_op_expr->indexed_obns().size());
    for (int i = 0; i < fw_op_expr->indexed_obns().size(); ++i) {
      const auto& obn = fw_op_expr->indexed_obns().at(i);
      const auto& lbi = fw_adapter_op->BnInOp2Lbi(obn);
      fw_output_index2lbi_symbol_.at(i) = lbi2lbi_symbol.at(lbi);
    }
    fw_output_index2grad_lbi_symbol_.resize(fw_op_expr->indexed_obns().size());
    for (int i = 0; i < fw_op_expr->indexed_obns().size(); ++i) {
      const auto& obn = fw_op_expr->indexed_obns().at(i);
      const auto& grad_lbi = obn2grad_lbi.at(obn);
      fw_output_index2grad_lbi_symbol_.at(i) = lbi2lbi_symbol.at(grad_lbi);
    }
  }
  // Updates fw_ibn_index_and_saved_index_pairs_ and fw_obn_index_and_saved_index_pairs_
  {
    HashSet<LogicalBlobId> fw_input_lbis;
    for (const auto& ibn : fw_adapter_op->input_bns()) {
      fw_input_lbis.insert(fw_adapter_op->BnInOp2Lbi(ibn));
    }
    HashSet<LogicalBlobId> fw_output_lbis;
    for (const auto& obn : fw_adapter_op->output_bns()) {
      fw_output_lbis.insert(fw_adapter_op->BnInOp2Lbi(obn));
    }
    HashSet<LogicalBlobId> captured_fw_input_lbis;
    HashSet<LogicalBlobId> captured_fw_output_lbis;
    const auto& UpdateCapturedLbis = [&](const LogicalBlobId& lbi) {
      if (fw_input_lbis.count(lbi) > 0) { captured_fw_input_lbis.insert(lbi); }
      if (fw_output_lbis.count(lbi) > 0) { captured_fw_output_lbis.insert(lbi); }
    };
    for (const auto& pair : ibn2grad_lbi) { UpdateCapturedLbis(pair.second); }
    for (const auto& bw_op : bw_adapter_ops) {
      for (const auto& ibn : bw_op->input_bns()) { UpdateCapturedLbis(bw_op->BnInOp2Lbi(ibn)); }
      for (const auto& obn : bw_op->output_bns()) { UpdateCapturedLbis(bw_op->BnInOp2Lbi(obn)); }
    }
    int captured_index = 0;
    for (int input_index = 0; input_index < fw_op_expr->indexed_ibns().size(); ++input_index) {
      const auto& ibn = fw_op_expr->indexed_ibns().at(input_index);
      if (captured_fw_input_lbis.count(fw_adapter_op->BnInOp2Lbi(ibn)) > 0) {
        fw_ibn_index_and_saved_index_pairs_.push_back(
            std::make_pair(input_index, captured_index++));
      }
    }
    for (int output_index = 0; output_index < fw_op_expr->indexed_obns().size(); ++output_index) {
      const auto& obn = fw_op_expr->indexed_obns().at(output_index);
      if (captured_fw_output_lbis.count(fw_adapter_op->BnInOp2Lbi(obn)) > 0) {
        fw_obn_index_and_saved_index_pairs_.push_back(
            std::make_pair(output_index, captured_index++));
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DefaultOpExprGradFunction::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                                               const TensorTuple& outputs,
                                               const AttrValueMap& attrs) const {
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
  std::vector<std::shared_ptr<one::Tensor>> lbi_symbol2tensor(lbi_symbol2lbi_.size());
  const auto& saved_tensors = ctx->SavedTensors();
  // Fills lbi_symbol2tensor by captured forward inputs
  for (const auto& pair : fw_ibn_index_and_saved_index_pairs_) {
    int lbi_symbol = fw_input_index2lbi_symbol_.at(pair.first);
    CHECK(!static_cast<bool>(lbi_symbol2tensor.at(lbi_symbol)));
    lbi_symbol2tensor.at(lbi_symbol) = saved_tensors.at(pair.second);
  }
  // Fills lbi_symbol2tensor by captured forward outputs
  for (const auto& pair : fw_obn_index_and_saved_index_pairs_) {
    int lbi_symbol = fw_output_index2lbi_symbol_.at(pair.first);
    CHECK(!static_cast<bool>(lbi_symbol2tensor.at(lbi_symbol)));
    lbi_symbol2tensor.at(lbi_symbol) = saved_tensors.at(pair.second);
  }
  // Fills lbi_symbol2tensor by backward out_grads
  for (int i = 0; i < fw_output_index2grad_lbi_symbol_.size(); ++i) {
    int lbi_symbol = fw_output_index2grad_lbi_symbol_.at(i);
    CHECK(!static_cast<bool>(lbi_symbol2tensor.at(lbi_symbol)));
    lbi_symbol2tensor.at(lbi_symbol) = out_grads.at(i);
  }
  for (const auto& entry : backward_entries_) {
    const auto& bw_input_index2lbi_symbol = entry.bw_input_index2lbi_symbol;
    // Initiates inputs
    TensorTuple inputs(bw_input_index2lbi_symbol.size());
    for (int i = 0; i < bw_input_index2lbi_symbol.size(); ++i) {
      inputs.at(i) = lbi_symbol2tensor.at(bw_input_index2lbi_symbol.at(i));
      CHECK(static_cast<bool>(inputs.at(i)));
    }
    const auto& bw_op_expr = entry.bw_op_expr;
    const auto& bw_output_index2lbi_symbol = entry.bw_output_index2lbi_symbol;
    TensorTuple outputs(bw_output_index2lbi_symbol.size());
    const auto& interpreter = JUST(OpInterpUtil::GetInterpreter());
    JUST(interpreter->Apply(*bw_op_expr, inputs, &outputs));
    // Update lbi_symbol2tensor by results
    for (int i = 0; i < bw_output_index2lbi_symbol.size(); ++i) {
      CHECK(static_cast<bool>(outputs.at(i)));
      lbi_symbol2tensor.at(bw_output_index2lbi_symbol.at(i)) = outputs.at(i);
    }
  }
  // Updates the result in_grads
  in_grads->resize(fw_input_index2grad_lbi_symbol_.size());
  for (int i = 0; i < fw_input_index2grad_lbi_symbol_.size(); ++i) {
    in_grads->at(i) = lbi_symbol2tensor.at(fw_input_index2grad_lbi_symbol_.at(i));
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("default", DefaultOpExprGradFunction);

}  // namespace one
}  // namespace oneflow
