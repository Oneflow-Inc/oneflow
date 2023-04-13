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

#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_CACHECONTEXT_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_CACHECONTEXT_H_

#include "OneFlow/OKL/Kernel/RegContext.h"
#include "oneflow/core/common/tensor_desc.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace okl {

class InitContext final : public user_op::KernelCacheContext, public user_op::KernelInitContext {
 public:
  InitContext(RegContext const* reg_ctx, user_op::KernelComputeContext* compute_ctx)
      : reg_ctx_(reg_ctx), compute_ctx_(compute_ctx) {}

  DeviceType device_type() const override { return reg_ctx_->device_type(); }
  const ParallelContext& parallel_ctx() const override { return compute_ctx_->parallel_ctx(); }
  ep::Stream* stream() override { return compute_ctx_->stream(); }

  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    return reg_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
  }

  const SbpParallel& SbpParallel4ArgNameAndIndex(const std::string&, int32_t) const override {
    TODO();
  }
  const user_op::TensorDesc* LogicalTensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                               int32_t index) const override {
    return reg_ctx_->TensorDesc4ArgNameAndIndex(arg_name, index);
  }
  const ParallelDesc& parallel_desc() const override { TODO(); }
  const NdSbp& NdSbp4ArgNameAndIndex(const std::string&, int32_t) const override { TODO(); }

  const std::vector<std::pair<std::string, int32_t>>& inputs() const override {
    return reg_ctx_->inputs();
  }
  const std::vector<std::pair<std::string, int32_t>>& outputs() const override {
    return reg_ctx_->outputs();
  }

 private:
  RegContext const* reg_ctx_;
  user_op::KernelComputeContext* compute_ctx_;

  const user_op::UserOpConfWrapper& user_op_conf() const override {
    return reg_ctx_->user_op_conf();
  }
  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return reg_ctx_->Attr4Name(attr_name);
  }
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_OKL_KERNEL_CACHECONTEXT_H_
