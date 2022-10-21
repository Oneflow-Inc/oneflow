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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_REGCONTEXT_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_REGCONTEXT_H_

#include <string>
#include <vector>
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "OneFlow/UserOpReflection.h"

namespace oneflow {
namespace okl {
// this context should support querying information about the kernel from representation in MLIR
using ArgVec = std::vector<std::pair<std::string, int32_t>>;
class RegContext final : public user_op::KernelRegContext {
 public:
  explicit RegContext(mlir::Operation* op) : op_(op) {
    for (const auto& operand_id : ::llvm::enumerate(
             mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedOperandSegments>(op))) {
      user_op::NaiveTensorDesc tensor_desc{};
      auto operand = op->getOperand(operand_id.index());
      if (auto rankedTensorType = operand.getType().dyn_cast<mlir::RankedTensorType>()) {
        tensor_desc.set_shape(
            Shape{rankedTensorType.getShape().begin(), rankedTensorType.getShape().end()});
        tensor_desc.set_data_type(
            mlir::oneflow::support::GetDataTypeFromMLIRType(rankedTensorType.getElementType()));
        // TODO: set stride
        // TODO: set is_dynamic
      } else {
        LOG(FATAL) << "Unranked tensor type not supported";
      }
      CHECK(arg2tensor_desc_.emplace(operand_id.value(), tensor_desc).second) << "duplicate key";
      inputs_.push_back(operand_id.value());
    }
    for (const auto& result_id : ::llvm::enumerate(
             ::mlir::oneflow::user_op::ArgIds<mlir::OpTrait::AttrSizedResultSegments>(op))) {
      user_op::NaiveTensorDesc tensor_desc{};
      auto result = op->getResult(result_id.index());
      if (auto rankedTensorType = result.getType().dyn_cast<mlir::RankedTensorType>()) {
        tensor_desc.set_shape(
            Shape{rankedTensorType.getShape().begin(), rankedTensorType.getShape().end()});
        tensor_desc.set_data_type(
            mlir::oneflow::support::GetDataTypeFromMLIRType(rankedTensorType.getElementType()));
        // TODO: set stride
        // TODO: set is_dynamic
      } else {
        LOG(FATAL) << "Unranked tensor type not supported";
      }
      CHECK(arg2tensor_desc_.emplace(result_id.value(), tensor_desc).second) << "duplicate key";
      outputs_.push_back(result_id.value());
    }
    auto dev_tag = mlir::OpTrait::IsOpConfCompatible<void>::getDeviceTag(op);
    if (dev_tag == "cpu") {
      device_type_ = DeviceType::kCPU;
    } else if (dev_tag == "cuda") {
      device_type_ = DeviceType::kCUDA;
    } else {
      LOG(FATAL) << "Unsupported device tag: " << dev_tag.str();
    }
  }

  ~RegContext() = default;
  DeviceType device_type() const override { return device_type_; }
  const ParallelContext& parallel_ctx() const override {
    TODO() << "create parallel_ctx from op in mlir";
    ParallelContext* parallel_ctx = nullptr;
    return *parallel_ctx;
  }
  const user_op::TensorDesc* TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                        int32_t index) const override {
    auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
    if (it == arg2tensor_desc_.end()) {
      LOG(FATAL) << "TensorDesc not found for arg_name: " << arg_name << " index: " << index;
    }
    return &(it->second);
  }
  const ArgVec& inputs() const override { return inputs_; }
  const ArgVec& outputs() const override { return outputs_; }

  const user_op::UserOpConfWrapper& user_op_conf() const override {
    TODO() << "get user op conf from op in mlir";
    OperatorConf user_op_conf;
    return user_op::UserOpConfWrapper(std::make_shared<OperatorConf>(user_op_conf));
  }

  const std::shared_ptr<const user_op::AttrVal>& Attr4Name(
      const std::string& attr_name) const override {
    return user_op_conf().Attr4Name(attr_name);
  }

  ::mlir::Operation* GetOp() const { return op_; }

 private:
  ::mlir::Operation* op_;
  DeviceType device_type_ = DeviceType::kInvalidDevice;
  std::unordered_map<mlir::oneflow::user_op::ArgID, user_op::NaiveTensorDesc> arg2tensor_desc_{};
  ArgVec inputs_;
  ArgVec outputs_;
};

}  // namespace okl
}  // namespace oneflow

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_KERNEL_LAUNCH_REGCONTEXT_H_
