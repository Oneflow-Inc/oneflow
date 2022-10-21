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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_RUNCONTEXT_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_RUNCONTEXT_H_

#include "oneflow/ir/oneflow-extension/include/OneFlow/kernel_launch/RegContext.h"

namespace oneflow {
namespace okl {
class RunContext final : public user_op::KernelComputeContext {
 public:
  explicit RunContext(std::shared_ptr<RegContext> reg, user_op::KernelComputeContext* comp)
      : reg_ctx_(std::move(reg)), comp_ctx_(comp) {}
  ~RunContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return reg_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override {
    auto op = reg_ctx_->GetOp();
    auto id = std::make_pair(arg_name, index);
    for (const auto& operand_id : ::llvm::enumerate(
             ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedOperandSegments>(op))) {
      if (operand_id.value() == id) {
        if (auto arg = op->getOperand(operand_id.index()).dyn_cast<mlir::BlockArgument>()) {
          return comp_ctx_->Tensor4ArgNameAndIndex("in", arg.getArgNumber());
        }
      }
    }
    for (const auto& result_id : ::llvm::enumerate(
             ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedResultSegments>(op))) {
      if (result_id.value() == id) {
        auto value = op->getResult(result_id.index());
        if (value.hasOneUse()) {
          mlir::Operation* first_user = value.use_begin()->getOwner();
          if (auto ret = llvm::dyn_cast_or_null<mlir::func::ReturnOp>(first_user)) {
            return comp_ctx_->Tensor4ArgNameAndIndex("out",
                                                     value.getUses().begin()->getOperandNumber());
          }
        }
      }
    }
    LOG(FATAL) << "Not supported";
  }

  ep::Stream* stream() override { return comp_ctx_->stream(); }

  DeviceType device_type() const override { return reg_ctx_->device_type(); }
  const ParallelContext& parallel_ctx() const override { return comp_ctx_->parallel_ctx(); }

  const ArgVec& inputs() const override { return reg_ctx_->inputs(); }
  const ArgVec& outputs() const override { return reg_ctx_->outputs(); }

  const user_op::UserOpConfWrapper& user_op_conf() const override {
    return reg_ctx_->user_op_conf();
  }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }
  std::shared_ptr<RegContext> reg_ctx_;
  KernelComputeContext* comp_ctx_ = nullptr;
  std::unordered_map<mlir::oneflow::user_op::ArgID, user_op::Tensor*> tensor_desc_{};
};
}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_RUNCONTEXT_H_
