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

#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_INFERCONTEXT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_INFERCONTEXT_H_

#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/kernel/user_kernel.h"
#include "OneFlow/OKL/Kernel/RegContext.h"

namespace oneflow {
namespace okl {

class InferContext final : public user_op::InferContext {
 public:
  explicit InferContext(RegContext const* reg_ctx);

  const user_op::TensorDesc& InputTensorDesc(const std::string& arg_name,
                                             int32_t index) const override {
    return *LogicalTensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const user_op::TensorDesc& OutputTensorDesc(const std::string& arg_name,
                                              int32_t index) const override {
    return *LogicalTensorDesc4ArgNameAndIndex(arg_name, index);
  }
  user_op::TensorDesc* MutOutputTensorDesc(const std::string&, int32_t) override { TODO(); }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override;

  const Shape& InputShape(const std::string& arg_name, int32_t index) const override;

  const Shape& OutputShape(const std::string&, int32_t) const override { TODO(); }
  void SetOutputShape(const std::string&, int32_t, const Shape&) override { TODO(); }
  const Shape& Shape4ArgNameAndIndex(const std::string& arg_name, int32_t index) const override;
  void SetShape4ArgNameAndIndex(const std::string&, int32_t, const Shape&) override { TODO(); }
  const Stride& InputStride(const std::string&, int32_t) const override { TODO(); }
  const Stride& OutputStride(const std::string&, int32_t) const override { TODO(); }
  void SetOutputStride(const std::string&, int32_t, const Stride&) override { TODO(); }
  const Stride& Stride4ArgNameAndIndex(const std::string&, int32_t) const override { TODO(); }
  void SetStride4ArgNameAndIndex(const std::string&, int32_t, const Stride&) override { TODO(); }
  DataType InputDType(const std::string&, int32_t) const override { TODO(); }
  DataType OutputDType(const std::string&, int32_t) const override { TODO(); }
  void SetOutputDType(const std::string&, int32_t, DataType) override { TODO(); }
  DataType Dtype4ArgNameAndIndex(const std::string&, int32_t) const override { TODO(); }
  void SetDtype4ArgNameAndIndex(const std::string&, int32_t, DataType) override { TODO(); }

  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    return reg_ctx_->inputs();
  }
  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    return reg_ctx_->outputs();
  }

  const std::string& input(const std::string& arg_name, int32_t index) const override {
    return reg_ctx_->user_op_conf().input(arg_name, index);
  }
  const std::string& output(const std::string& arg_name, int32_t index) const override {
    return reg_ctx_->user_op_conf().output(arg_name, index);
  }

  bool has_input(const std::string& arg_name, int32_t index) const override {
    return reg_ctx_->user_op_conf().has_input(arg_name, index);
  }
  bool has_output(const std::string& arg_name, int32_t index) const override {
    return reg_ctx_->user_op_conf().has_input(arg_name, index);
  }

  int32_t input_size(const std::string& arg_name) const override {
    return reg_ctx_->user_op_conf().input_size(arg_name);
  }
  int32_t output_size(const std::string& arg_name) const override {
    return reg_ctx_->user_op_conf().output_size(arg_name);
  }
  const std::string& op_name() const override { return reg_ctx_->user_op_conf().op_name(); }
  const std::string& op_type_name() const override {
    return reg_ctx_->user_op_conf().op_type_name();
  }
  const std::string& op_loc() const override { TODO(); }

  const ParallelContext& parallel_ctx() const override { TODO(); }
  const ParallelDesc& parallel_desc() const override { TODO(); }

  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&, int32_t) const override {
    TODO();
  }

  const NdSbp& NdSbp4ArgNameAndIndex(const std::string&, int32_t) const override { TODO(); }

  bool InputIsDynamic(const std::string&, int32_t) const override { TODO(); }
  bool OutputIsDynamic(const std::string&, int32_t) const override { TODO(); }
  void SetOutputIsDynamic(const std::string&, int32_t, bool) override { TODO(); }
  bool IsDynamic4ArgNameAndIndex(const std::string&, int32_t) const override { TODO(); }
  void SetIsDynamic4ArgNameAndIndex(const std::string&, int32_t, bool) override { TODO(); }

  int64_t parallel_num() const override { TODO(); }

 private:
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override;

  RegContext const* reg_ctx_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_INFERCONTEXT_H_