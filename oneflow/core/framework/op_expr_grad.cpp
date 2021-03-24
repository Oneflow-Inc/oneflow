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

#include "oneflow/core/framework/op_expr_grad.h"

#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter_util.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/user_op.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {
namespace one {

class MatMulOpGrad : public OpExprGrad {
 public:
  explicit MatMulOpGrad(const OpExpr& op) : OpExprGrad(op) {
    // TODO()
    UNIMPLEMENTED();
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {
    CHECK_EQ_OR_RETURN(inputs.size(), 2);
    if (a_grad_op_.get()) { ctx->SaveTensorForBackward(inputs[1]); }
    if (b_grad_op_.get()) { ctx->SaveTensorForBackward(inputs[0]); }
    return Maybe<void>::Ok();
  }

  Maybe<void> DoBackward(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const override {
    CHECK_EQ_OR_RETURN(out_grads.size(), 1);
    const auto& saved_tensors = ctx->SavedTensors();
    in_grads->resize(2);
    const auto& interpreter = JUST(OpInterpUtil::GetInterpreter());
    int i = 0;
    if (a_grad_op_.get()) {
      TensorTuple inputs(2);
      inputs.push_back(out_grads[0]);
      inputs.push_back(saved_tensors[i]);
      TensorTuple outputs(1);

      JUST(interpreter->Apply(*a_grad_op_, inputs, &outputs));
      (*in_grads)[0] = outputs[0];
      ++i;
    }
    if (b_grad_op_.get()) {
      TensorTuple inputs(2);
      inputs.push_back(out_grads[0]);
      inputs.push_back(saved_tensors[i]);
      TensorTuple outputs(1);

      JUST(interpreter->Apply(*b_grad_op_, inputs, &outputs));
      (*in_grads)[1] = outputs[0];
    }
    return Maybe<void>::Ok();
  }

 private:
  std::shared_ptr<OpExpr> a_grad_op_;
  std::shared_ptr<OpExpr> b_grad_op_;
};

REGISTER_OP_EXPR_GRAD("matmul", MatMulOpGrad);

namespace {
void GetUserOpGradConf(const UserOpExpr& fw_op_expr, std::vector<OperatorConf>* bw_op_confs) {
  OperatorConf op_conf;
  fw_op_expr.BuildOpConf(&op_conf);

  UserOp op_adapter;
  op_adapter.Init(op_conf);

  HashMap<std::string, LogicalBlobId> in_bn2lbi;
  HashMap<std::string, LogicalBlobId> out_bn2lbi;
  auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
    if (std::find(fw_op_expr.indexed_ibns().begin(), fw_op_expr.indexed_ibns().end(), bn)
        != fw_op_expr.indexed_ibns().end()) {
      return &in_bn2lbi[bn];
    } else if (std::find(fw_op_expr.indexed_obns().begin(), fw_op_expr.indexed_obns().end(), bn)
               != fw_op_expr.indexed_obns().end()) {
      return &out_bn2lbi[bn];
    } else {
      return nullptr;
    }
  };

  const auto& dummy_blob_desc = BlobDesc(Shape(), DataType::kInvalidDataType);
  auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
    return dummy_blob_desc;
  };

  const auto& op_type_case = op_conf.op_type_case();
  CHECK((IsClassRegistered<int32_t, GenerateBackwardOpConfWrapperStruct>(op_type_case)))
      << PbMessage2TxtString(op_conf);
  std::unique_ptr<GenerateBackwardOpConfWrapperStruct> obj;
  obj.reset(NewObj<int32_t, GenerateBackwardOpConfWrapperStruct>(op_type_case));
  obj->Call(op_adapter, bw_op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp);
}
}  // namespace

class UserOpExprGrad : public OpExprGrad {
 public:
  explicit UserOpExprGrad(const OpExpr& op) : OpExprGrad(op) {
    const auto& fw_op_expr = dynamic_cast<const UserOpExpr&>(op);
    std::vector<OperatorConf> bw_op_confs;
    GetUserOpGradConf(fw_op_expr, &bw_op_confs);
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {
    // TODO()
    UNIMPLEMENTED();
    return Maybe<void>::Ok();
  }

  Maybe<void> DoBackward(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const override {
    // TODO()
    UNIMPLEMENTED();
    return Maybe<void>::Ok();
  }

 private:
  std::vector<std::shared_ptr<OpExpr>> backward_ops_;
};

REGISTER_OP_EXPR_GRAD("user_op", UserOpExprGrad);
}  // namespace one
}  // namespace oneflow
