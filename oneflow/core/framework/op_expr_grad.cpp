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
namespace {
Maybe<void> GetUserOpGradConf(const UserOpExpr* fw_op_expr, std::vector<OperatorConf>* bw_op_confs) {
  OperatorConf op_conf;
  fw_op_expr->BuildOpConf(&op_conf);

  UserOp op_adapter;
  op_conf.set_device_tag("cpu");
  op_adapter.Init(op_conf);

  std::cout << "fw_op_expr op_name " << fw_op_expr->op_name() << std::endl;
  for (const auto& in_bn : fw_op_expr->indexed_ibns()) {
    std::cout << "fw_op_expr ibn " << in_bn << std::endl;
  }
  for (const auto& out_bn : fw_op_expr->indexed_obns()) {
    std::cout << "fw_op_expr obn " << out_bn << std::endl;
  }
  HashMap<std::string, LogicalBlobId> in_rbn2diff_lbi;
  HashMap<std::string, LogicalBlobId> out_rbn2diff_lbi;
  auto DiffLbi4BnInOp = [&](const std::string& rbn) -> LogicalBlobId* {
    if (std::find(fw_op_expr->indexed_ibns().begin(), fw_op_expr->indexed_ibns().end(), rbn)
        != fw_op_expr->indexed_ibns().end()) {
      return &in_rbn2diff_lbi[rbn];
    } else if (std::find(fw_op_expr->indexed_obns().begin(), fw_op_expr->indexed_obns().end(), rbn)
               != fw_op_expr->indexed_obns().end()) {
      auto find_iter = out_rbn2diff_lbi.find(rbn);
      if (find_iter == out_rbn2diff_lbi.end()) {
        LogicalBlobId lbi;
        lbi.set_op_name("next_grad_op");
        lbi.set_blob_name(rbn + "_diff");
        out_rbn2diff_lbi.emplace(rbn, lbi);
      }
      return &out_rbn2diff_lbi[rbn];
    } else {
      return nullptr;
    }
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
  for(const auto& pair : in_rbn2diff_lbi) {
    std::cout << "in bn " << pair.first << " diff lib " << pair.second.DebugString() << std::endl;
  }
  for(const auto& pair : out_rbn2diff_lbi) {
    std::cout << "out bn " << pair.first << " diff lib " << pair.second.DebugString() << std::endl;
  }
  return Maybe<void>::Ok();
}
}  // namespace

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
                      const TensorTuple& outputs) const override {}

  Maybe<void> DoBackward(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const override {}

 private:
  std::vector<std::shared_ptr<OpExpr>> backward_ops_;
};

REGISTER_OP_EXPR_GRAD("UserOp", UserOpExprGrad);
}  // namespace one
}  // namespace oneflow
