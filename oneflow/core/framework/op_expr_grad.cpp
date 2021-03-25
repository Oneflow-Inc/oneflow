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

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/framework/op_interpreter_util.h"
#include "oneflow/core/job_rewriter/autograd.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/operator/operator.h"
#include "oneflow/core/operator/user_op.h"
#include "oneflow/core/register/logical_blob_id.pb.h"

namespace oneflow {
namespace one {

class UserOpExprGrad : public OpExprGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* fw_op_expr = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(fw_op_expr);
    std::vector<OperatorConf> bw_op_confs;
    JUST(GetUserOpGradConf(fw_op_expr, &bw_op_confs));
    for (const auto& op_conf : bw_op_confs) {
      std::cout << op_conf.DebugString() << std::endl;
      // backward_ops.emplace_back(std::make_shared<OpExpr>(op_conf));
    }
    return Maybe<void>::Ok();
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {
    return Maybe<void>::Ok();
  }

  Maybe<void> DoBackward(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const override {
    return Maybe<void>::Ok();
  }

  Maybe<void> GetUserOpGradConf(const UserOpExpr* fw_op_expr,
                                std::vector<OperatorConf>* bw_op_confs);

 private:
  std::vector<std::shared_ptr<OpExpr>> backward_ops_;

  HashMap<std::string, LogicalBlobId> in_bn2diff_lbi_;
  HashMap<std::string, LogicalBlobId> out_bn2diff_lbi_;
};

Maybe<void> UserOpExprGrad::GetUserOpGradConf(const UserOpExpr* fw_op_expr,
                                              std::vector<OperatorConf>* bw_op_confs) {
  OperatorConf op_conf;
  fw_op_expr->BuildOpConf(&op_conf);
  op_conf.set_device_tag("cpu");
  UserOp op_adapter;
  op_adapter.Init(op_conf);

  auto DiffLbi4BnInOp = [&](const std::string& bn) -> LogicalBlobId* {
    if (std::find(fw_op_expr->indexed_ibns().begin(), fw_op_expr->indexed_ibns().end(), bn)
        != fw_op_expr->indexed_ibns().end()) {
      return &in_bn2diff_lbi_[bn];
    } else if (std::find(fw_op_expr->indexed_obns().begin(), fw_op_expr->indexed_obns().end(), bn)
               != fw_op_expr->indexed_obns().end()) {
      auto it = out_bn2diff_lbi_.find(bn);
      if (it == out_bn2diff_lbi_.end()) {
        LogicalBlobId lbi;
        lbi.set_op_name("_");
        lbi.set_blob_name(bn);
        it = out_bn2diff_lbi_.emplace(bn, lbi).first;
      }
      return &(it->second);
    } else {
      LOG(FATAL) << "diff lbi for bn in op not found, bn: " << fw_op_expr->op_name() << "/" << bn;
    }
    return nullptr;
  };
  const auto& dummy_blob_desc = BlobDesc(Shape(), DataType::kInvalidDataType);
  auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
    return dummy_blob_desc;
  };

  const auto& op_type_case = op_conf.op_type_case();
  CHECK_OR_RETURN((IsClassRegistered<int32_t, GenerateBackwardOpConfWrapperStruct>(op_type_case)));
  std::unique_ptr<GenerateBackwardOpConfWrapperStruct> obj;
  obj.reset(NewObj<int32_t, GenerateBackwardOpConfWrapperStruct>(op_type_case));
  JUST(obj->Call(op_adapter, bw_op_confs, DiffLbi4BnInOp, LogicalBlobDesc4BnInOp));
  return Maybe<void>::Ok();
}

REGISTER_OP_EXPR_GRAD("matmul", UserOpExprGrad);
}  // namespace one
}  // namespace oneflow
