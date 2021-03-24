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
#include "oneflow/core/framework/op_interpreter_util.h"
#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {
namespace one {

class UserOpGrad : public OpExprGrad {
 public:
  Maybe<void> Init(const OpExpr& op) override {
    const auto* user_op = dynamic_cast<const UserOpExpr*>(&op);
    CHECK_NOTNULL_OR_RETURN(user_op);
    const UserOpConf& user_conf = user_op->proto();
    const user_op::OpGradRegistryResult* val =
      user_op::UserOpRegistryMgr::Get().GetOpGradRegistryResult(user_conf.op_type_name());
    if (val == nullptr) {
      return Error::GradientFunctionNotFound() << user_conf.op_type_name();
    }

    std::vector<OperatorConf> bw_op_confs;
    BlobDesc fake_blob_desc(DataType::kFloat);
    auto LogicalBlobDesc4BnInOp = [&](const std::string& bn) -> const BlobDesc& {
      return fake_blob_desc;
    };
    HashMap<std::string, LogicalBlobId> in_diff_lbis;
    auto DiffLbi4BnInOp = [&](const std::string& bn) {
      return &in_diff_lbis[bn];
    };
    OperatorConf op_conf;
    user_op->BuildOpConf(&op_conf);
    user_op::UserOpWrapper fw_user_op(op_conf, LogicalBlobDesc4BnInOp, DiffLbi4BnInOp);
    if (nullptr != val->bw_gen_fn) {
      // new refined interface
      user_op::BackwardOpConfContext ctx(fw_user_op, &bw_op_confs);
      val->bw_gen_fn(&ctx);
    } else if (nullptr != val->gen_bw_fn) {
      // old interface, will be removed when all backward gradient configs are using new interface
      auto AddOp = [&](const user_op::UserOpConfWrapper& wrapper) {
        bw_op_confs.push_back(wrapper.op_conf());
      };
      val->gen_bw_fn(fw_user_op, AddOp);
    }

    for (const auto& op_conf : bw_op_confs) {
      // TODO()
    }
  }

  Maybe<void> Capture(OpExprInterpState* ctx, const TensorTuple& inputs,
                      const TensorTuple& outputs) const override {}

  Maybe<void> DoBackward(const OpExprInterpState* ctx, const TensorTuple& out_grads,
                         TensorTuple* in_grads) const override {}

 private:
  std::vector<std::shared_ptr<OpExpr>> backward_ops_;
};

REGISTER_OP_EXPR_GRAD("UserOp", UserOpGrad);

}  // namespace one
}  // namespace oneflow
