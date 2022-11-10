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
#include "OneFlow/UserOpReflection.h"
#include "oneflow/core/framework/infer_util.h"
#include "oneflow/core/framework/user_op_attr.pb.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/kernel/blob_tensor_view.h"
#include "oneflow/core/memory/memory_case.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/kernel_launch/InferContext.h"
#include "oneflow/ir/oneflow-extension/include/OneFlow/kernel_launch/RegContext.h"
#include "oneflow/core/framework/user_op_kernel_registry.h"
#include "oneflow/ir/oneflow-translate/include/OneFlow/MLIROneFlowTranslation.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/Support/LogicalResult.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/Operation.h"

#include <memory>
#include <string>
#include <vector>

namespace oneflow {
namespace okl {

static user_op::UserOpConfWrapper GetConfWrapper(mlir::Operation* op) {
  OperatorConf op_conf;
  if (mlir::failed(mlir::oneflow::ConvertUserOpAttributes(op, op_conf))) {
    op->emitError("fail to convert user op attributes");
    exit(1);
  }
  auto conf_wrapper_ = user_op::UserOpConfWrapper(std::make_shared<OperatorConf>(op_conf));
  return conf_wrapper_;
}

RegContext::RegContext(mlir::Operation* op) : op_(op), conf_wrapper_(GetConfWrapper(op)) {
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

DeviceType RegContext::device_type() const { return device_type_; }
const ParallelContext& RegContext::parallel_ctx() const {
  TODO() << "create parallel_ctx from op in mlir";
  ParallelContext* parallel_ctx = nullptr;
  return *parallel_ctx;
}
const user_op::TensorDesc* RegContext::TensorDesc4ArgNameAndIndex(const std::string& arg_name,
                                                                  int32_t index) const {
  auto it = arg2tensor_desc_.find(std::make_pair(arg_name, index));
  if (it == arg2tensor_desc_.end()) {
    LOG(FATAL) << "TensorDesc not found for arg_name: " << arg_name << " index: " << index;
  }
  return &(it->second);
}
const ArgVec& RegContext::inputs() const { return inputs_; }
const ArgVec& RegContext::outputs() const { return outputs_; }

// TODO: more information is needed
const user_op::UserOpConfWrapper& RegContext::user_op_conf() const { return conf_wrapper_; }

const std::shared_ptr<const user_op::AttrVal>& RegContext::Attr4Name(
    const std::string& attr_name) const {
  return user_op_conf().Attr4Name(attr_name);
}

::mlir::Operation* RegContext::GetOp() const { return op_; }

const user_op::OpKernel* RegContext::GenKernel() {
  auto reg_res = CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
      GetOp()->getName().stripDialect().str(), *this));
  return reg_res->create_fn();
}

size_t RegContext::GetTmpBufferSize() {
  auto reg_res = CHECK_JUST(user_op::UserOpRegistryMgr::Get().GetOpKernelRegistryResult(
      GetOp()->getName().stripDialect().str(), *this));
  if (reg_res->need_temp_storage) {
    InferContext infer_ctx(this);
    return reg_res->infer_tmp_size_fn(&infer_ctx);
  }
  return 0;
}

}  // namespace okl
}  // namespace oneflow
