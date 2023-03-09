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
#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_REGCONTEXT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_REGCONTEXT_H_

#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "OneFlow/UserOpReflection.h"
#include "mlir/IR/Operation.h"

namespace oneflow {
namespace okl {
// this context should support querying information about the kernel from representation in MLIR
using ArgVec = std::vector<std::pair<std::string, int32_t>>;
class RegContext final : public user_op::KernelRegContext {
 public:
  explicit RegContext(mlir::Operation* op);
  ~RegContext() = default;

  // override user_op KernelRegContext
  DeviceType device_type() const override;
  const ParallelContext& parallel_ctx() const override;
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override;
  const ArgVec& inputs() const override;
  const ArgVec& outputs() const override;
  const user_op::UserOpConfWrapper& user_op_conf() const override;
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override;

  const size_t GetTmpBufferSize() const;
  ::mlir::Operation* GetOp() const { return op_; };
  const user_op::OpKernel* GetKernel() const { return kernel_; };

 private:
  ::mlir::Operation* op_;
  DeviceType device_type_ = DeviceType::kInvalidDevice;
  std::unordered_map<mlir::oneflow::user_op::ArgID, user_op::NaiveTensorDesc> arg2tensor_desc_{};
  ArgVec inputs_;
  ArgVec outputs_;
  user_op::UserOpConfWrapper conf_wrapper_;

  const user_op::OpKernelRegistryResult* reg_res_ = nullptr;
  const user_op::OpKernel* kernel_ = nullptr;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_REGCONTEXT_H_
