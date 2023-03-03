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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_COMPUTECONTEXT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_COMPUTECONTEXT_H_

#include "mlir/IR/BuiltinAttributes.h"
#include "OneFlow/OKL/Kernel/RegContext.h"
#include "OneFlow/OKL/Kernel/TmpBufferManager.h"

namespace oneflow {
namespace okl {
class ComputeContext final : public user_op::KernelComputeContext {
 public:
  ComputeContext(RegContext const* reg_ctx, user_op::KernelComputeContext* comp_ctx)
      : reg_ctx_(reg_ctx),
        comp_ctx_(comp_ctx),
        tmp_buffer_(comp_ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0)) {}

  ~ComputeContext() = default;

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return reg_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  ep::Stream* stream() override { return comp_ctx_->stream(); }

  DeviceType device_type() const override { return reg_ctx_->device_type(); }
  const ParallelContext& parallel_ctx() const override { return comp_ctx_->parallel_ctx(); }

  const ArgVec& inputs() const override { return reg_ctx_->inputs(); }
  const ArgVec& outputs() const override { return reg_ctx_->outputs(); }

  const user_op::UserOpConfWrapper& user_op_conf() const override {
    return reg_ctx_->user_op_conf();
  }
  user_op::Tensor* Tensor4ArgNameAndIndex(const std::string& arg_name, int32_t index) override;

 private:
  RegContext const* reg_ctx_;
  KernelComputeContext* comp_ctx_;
  TmpBufferManager tmp_buffer_;

  std::unordered_map<mlir::oneflow::user_op::ArgID, user_op::Tensor*> tensor_{};

  user_op::Tensor* CreateTensorWithArgNameAndIndex(const std::string& arg_name, int32_t index);
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_COMPUTECONTEXT_H_
