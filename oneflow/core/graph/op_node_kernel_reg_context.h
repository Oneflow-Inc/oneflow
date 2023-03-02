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
#ifndef ONEFLOW_CORE_GRAPH_OP_NODE_KERNEL_REG_CONTEXT_H_
#define ONEFLOW_CORE_GRAPH_OP_NODE_KERNEL_REG_CONTEXT_H_

#include "oneflow/core/framework/user_op_registry_manager.h"

namespace oneflow {

class OpNode;

class OpNodeKernelRegContext final : public user_op::KernelRegContext {
 public:
  using ArgVec = std::vector<std::pair<std::string, int32_t>>;

  explicit OpNodeKernelRegContext(const OpNode* op_node);
  ~OpNodeKernelRegContext() = default;

  DeviceType device_type() const override { return device_type_; }

  const ParallelContext& parallel_ctx() const override { PRINT_BUG_PROMPT_AND_ABORT(); }
  int64_t parallel_num() const override { return parallel_num_; }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override;
  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }

  const user_op::UserOpConfWrapper& user_op_conf() const override { return user_op_conf_; }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override;

 private:
  const user_op::UserOpConfWrapper user_op_conf_;
  ArgVec inputs_;
  ArgVec outputs_;
  DeviceType device_type_;
  int64_t parallel_num_;
  HashMap<std::pair<std::string, int32_t>, user_op::NaiveTensorDesc> arg2tensor_desc_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_GRAPH_OP_NODE_KERNEL_REG_CONTEXT_H_
