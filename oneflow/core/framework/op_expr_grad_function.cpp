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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter_util.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {
namespace one {

class DefaultOpExprGradFunction : public OpExprGradFunction {
 public:
  // The snapshot indicates the indices of the required forward inputs and outputs
  // by each backward operator.
  struct Snapshot {
    std::vector<int> input_indices;
    std::vector<int> output_indices;
    int count = 0;
  };

  Maybe<void> Init(const OpExpr& op) override;

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override;

  Maybe<void> Apply(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                    TensorTuple* in_grads) const override;

 private:
  Maybe<void> GenerateOpGradConf(const Operator& op, std::vector<OperatorConf>* bw_op_confs);

  Maybe<void> UpdateRequiresBackward(const TensorTuple& inputs) const;

 private:
  std::vector<std::shared_ptr<OpExpr>> backward_ops_;

  std::vector<std::vector<int>> in_grad_indices_;
  std::vector<std::vector<int>> out_grad_indices_;

  // Snapshot information for each backward op.
  std::vector<Snapshot> snapshots_;

  // Whether each backward operator needs to do backward or not.
  mutable std::vector<bool> requires_backward_;

  // The input gradient logical blob id for each forward input blob name which
  // needs backward.
  HashMap<std::string, LogicalBlobId> in_bn2grad_lbi_;
  // The output gradient logical blob id for each forward output blob name.
  HashMap<std::string, LogicalBlobId> out_bn2grad_lbi_;
};

namespace {

Maybe<void> ProcessOpInputs(const Operator& op,  // NOLINT
                            const HashMap<std::string, int>& input_lbn_indices,
                            const HashMap<std::string, int>& output_lbn_indices,
                            const HashMap<std::string, std::string>& grad_lbn2bn,
                            const HashMap<std::string, int>& obn_indices,
                            DefaultOpExprGradFunction::Snapshot* snapshot,
                            std::vector<int>* out_grad_indices,
                            std::vector<std::string>* indexed_ibns) {
  // Input blob names in the backward op will be divided into 3 types which
  // means the input comes from either forward inputs, or forward outputs, or
  // backward output gradients.
  std::vector<std::vector<std::string>> typed_ibns(3);
  snapshot->count = 0;
  for (const auto& bn : op.input_bns()) {
    std::string lbn = GenLogicalBlobName(op.BnInOp2Lbi(bn));
    if (input_lbn_indices.count(lbn)) {
      snapshot->input_indices.emplace_back(input_lbn_indices.at(lbn));
      snapshot->count += 1;
      typed_ibns[0].emplace_back(bn);
    } else if (output_lbn_indices.count(lbn)) {
      snapshot->output_indices.emplace_back(output_lbn_indices.at(lbn));
      snapshot->count += 1;
      typed_ibns[1].emplace_back(bn);
    } else {
      const auto& it = grad_lbn2bn.find(lbn);
      // Otherwise this input should be output gradient.
      CHECK_OR_RETURN(it != grad_lbn2bn.end());
      // The output gradient index is equal to the forward output index.
      out_grad_indices->emplace_back(obn_indices.at(it->second));
      typed_ibns[2].emplace_back(bn);
    }
  }
  for (const auto& ibns : typed_ibns) {
    for (const auto& v : ibns) { indexed_ibns->emplace_back(v); }
  }
  return Maybe<void>::Ok();
}

Maybe<void> ProcessOpOutputs(const Operator& op,
                             const HashMap<std::string, std::string>& grad_lbn2bn,
                             const HashMap<std::string, int>& ibn_indices,
                             std::vector<int>* in_grad_indices,
                             std::vector<std::string>* indexed_obns) {
  for (const auto& bn : op.output_bns()) {
    indexed_obns->emplace_back(bn);
    std::string lbn = GenLogicalBlobName(op.BnInOp2Lbi(bn));
    const auto& it = grad_lbn2bn.find(lbn);
    CHECK_OR_RETURN(it != grad_lbn2bn.end());
    // The input gradient index is equal to the foward input index.
    in_grad_indices->emplace_back(ibn_indices.at(it->second));
  }
  return Maybe<void>::Ok();
}

}  // namespace

Maybe<void> DefaultOpExprGradFunction::GenerateOpGradConf(const Operator& op,
                                                          std::vector<OperatorConf>* bw_op_confs) {
  auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
    const auto& input_bns = op.input_bns();
    const auto& output_bns = op.output_bns();
    if (std::find(input_bns.begin(), input_bns.end(), bn) != input_bns.end()) {
      return &in_bn2grad_lbi_[bn];
    } else if (std::find(output_bns.begin(), output_bns.end(), bn) != output_bns.end()) {
      auto it = out_bn2grad_lbi_.find(bn);
      if (it == out_bn2grad_lbi_.end()) {
        LogicalBlobId lbi;
        lbi.set_op_name("_");
        lbi.set_blob_name(bn);
        it = out_bn2grad_lbi_.emplace(bn, lbi).first;
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
  fw_op_expr->BuildOpConf(&fw_op_conf);

  // Generate backward operator conf for each input. The `LogicalBlobId` for
  // backward output gradient is dummy due to inaccessibility.
  std::vector<OperatorConf> bw_op_confs;
  std::shared_ptr<Operator> op_adapter = ConstructOp(fw_op_conf, DeviceType::kCPU);
  JUST(GenerateOpGradConf(*op_adapter, &bw_op_confs));

  CHECK_EQ_OR_RETURN(op.input_num(), in_bn2grad_lbi_.size())
      << "All inputs are considered to require gradients since the `requires_grad` does not known "
         "to us here.";
  if (bw_op_confs.empty()) { return Maybe<void>::Ok(); }
  in_grad_indices_.resize(bw_op_confs.size());
  out_grad_indices_.resize(bw_op_confs.size());
  snapshots_.resize(bw_op_confs.size());
  requires_backward_.resize(bw_op_confs.size());

  HashMap<std::string, int> input_lbn_indices;
  HashMap<std::string, int> output_lbn_indices;
  HashMap<std::string, int> ibn_indices;
  HashMap<std::string, int> obn_indices;
  for (int i = 0; i < fw_op_expr->indexed_ibns().size(); ++i) {
    const auto& bn = fw_op_expr->indexed_ibns().at(i);
    ibn_indices.emplace(bn, i);
    std::string lbn = GenLogicalBlobName(op_adapter->BnInOp2Lbi(bn));
    input_lbn_indices.emplace(lbn, i);
  }
  for (int i = 0; i < fw_op_expr->indexed_obns().size(); ++i) {
    const auto& bn = fw_op_expr->indexed_obns().at(i);
    obn_indices.emplace(bn, i);
    std::string lbn = GenLogicalBlobName(op_adapter->BnInOp2Lbi(bn));
    output_lbn_indices.emplace(lbn, i);
  }

  HashMap<std::string, std::string> grad_lbn2bn;
  for (const auto& bn2lbi : {in_bn2grad_lbi_, out_bn2grad_lbi_}) {
    for (const auto& it : bn2lbi) {
      std::string lbn = GenLogicalBlobName(it.second);
      grad_lbn2bn.emplace(lbn, it.first);
    }
  }
  for (int i = 0; i < bw_op_confs.size(); ++i) {
    const auto& op_conf = bw_op_confs.at(i);
    VLOG(10) << op_conf.DebugString() << std::endl;
    std::shared_ptr<Operator> bw_op_adapter = ConstructOp(op_conf, DeviceType::kCPU);
    std::vector<std::string> indexed_ibns, indexed_obns;

    ProcessOpInputs(*bw_op_adapter, input_lbn_indices, output_lbn_indices, grad_lbn2bn, obn_indices,
                    &snapshots_[i], &out_grad_indices_[i], &indexed_ibns);
    ProcessOpOutputs(*bw_op_adapter, grad_lbn2bn, ibn_indices, &in_grad_indices_[i], &indexed_obns);

    // Currently only user op is considered.
    if (op_conf.has_user_conf()) {
      UserOpConf user_conf(op_conf.user_conf());
      backward_ops_.emplace_back(std::make_shared<UserOpExpr>(op_conf.name(), std::move(user_conf),
                                                              indexed_ibns, indexed_obns));
    } else {
      // TODO()
      UNIMPLEMENTED();
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DefaultOpExprGradFunction::UpdateRequiresBackward(const TensorTuple& inputs) const {
  for (int i = 0; i < in_grad_indices_.size(); ++i) {
    requires_backward_[i] = false;
    for (const auto& j : in_grad_indices_.at(i)) {
      if (inputs.at(j)->requires_grad()) {
        requires_backward_[i] = true;
        break;
      }
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DefaultOpExprGradFunction::Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                                               const TensorTuple& outputs) const {
  JUST(UpdateRequiresBackward(inputs));
  for (int i = 0; i < backward_ops_.size(); ++i) {
    if (!requires_backward_.at(i)) { continue; }
    for (const int& index : snapshots_.at(i).input_indices) {
      ctx->SaveTensorForBackward(inputs.at(index));
    }
    for (const int& index : snapshots_.at(i).output_indices) {
      ctx->SaveTensorForBackward(outputs.at(index));
    }
  }
  return Maybe<void>::Ok();
}

Maybe<void> DefaultOpExprGradFunction::Apply(const OpExprInterpState* ctx,
                                             const TensorTuple& out_grads,
                                             TensorTuple* in_grads) const {
  int input_size = in_bn2grad_lbi_.size();
  in_grads->resize(input_size);
  const auto& saved_tensors = ctx->SavedTensors();
  int offset = 0;
  for (int i = 0; i < backward_ops_.size(); ++i) {
    if (!requires_backward_.at(i)) { continue; }
    TensorTuple inputs;
    for (int j = 0; j < snapshots_.at(i).count; ++j) {
      inputs.emplace_back(saved_tensors.at(offset + j));
    }
    for (const int& index : out_grad_indices_.at(i)) { inputs.emplace_back(out_grads.at(index)); }
    TensorTuple outputs(in_grad_indices_.at(i).size());
    const auto& interpreter = JUST(OpInterpUtil::GetInterpreter());
    JUST(interpreter->Apply(*(backward_ops_.at(i)), inputs, &outputs));

    int j = 0;
    for (const int& index : in_grad_indices_.at(i)) { in_grads->at(index) = outputs.at(j++); }
    offset += snapshots_.at(i).count;
  }
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD_FUNCTION("default", DefaultOpExprGradFunction);

}  // namespace one
}  // namespace oneflow
