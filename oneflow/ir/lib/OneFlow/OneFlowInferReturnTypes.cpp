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
#include "OneFlow/OneFlowOps.h"
#include "OneFlow/UserOpConversion.h"
#include "OneFlow/UserOpReflection.h"

namespace mlir {

namespace oneflow {

namespace {

std::unique_ptr<::oneflow::BlobDesc> GetBlobDescFromMlirTensorType(TensorType tensor_type) {
  auto dtype = ::oneflow::kInvalidDataType;
  if (tensor_type.getElementType().isF32()) {
    dtype = ::oneflow::kFloat;
  } else {
    tensor_type.dump();
    LOG(FATAL) << "fail to get BlobDesc from TensorType";
  }
  auto shape_from_mlir = new ::oneflow::Shape(llvm::SmallVector<int64_t, 4>(
      {tensor_type.getShape().begin(), tensor_type.getShape().end()}));
  return std::make_unique<::oneflow::BlobDesc>(*shape_from_mlir, dtype);
}

static auto MagicalOpName = "INFER_MAGICAL";
LogicalResult ConvertUserOp(llvm::StringRef op_type_name, ::oneflow::OperatorConf& op_conf,
                            ValueRange operands, DictionaryAttr attributes) {
  oneflow::ConfOpAdaptor conf_op_adaptor(operands, attributes);
  std::string op_name = MagicalOpName;
  CHECK(
      user_op::ConvertUserOpInputs(op_type_name, operands, attributes, op_conf.mutable_user_conf())
          .succeeded());
  // if (!succeeded(ConvertUserOpOutputs(op, op_name, user_conf))) {
  //   op->emitError("fail to convert user op outputs");
  //   return failure();
  // }
  if (!succeeded(user_op::ConvertUserOpAttributes(op_type_name, operands, attributes, op_conf))) {
    return failure();
  }
  return success();
}

}  // namespace

::mlir::LogicalResult NormalizationAddReluOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  ::oneflow::OperatorConf op_conf{};
  const auto op_type_name = "normalization_add_relu";
  CHECK(ConvertUserOp(op_type_name, op_conf, operands, attributes).succeeded());
  auto op = CHECK_JUST(ConstructOp(op_conf));
  std::unordered_map<std::string, std::unique_ptr<::oneflow::BlobDesc>> lbi2logical_blob_desc_;
  LOG(ERROR) << op->op_conf().DebugString();

  auto operand_ids =
      user_op::ArgIds<OpTrait::AttrSizedOperandSegments>(op_type_name, operands, attributes);

  auto operand_index = 0;
  for (const auto& pair : operand_ids) {
    const auto& arg_name = pair.first;
    const auto& arg_num = pair.second;
    for (size_t i = 0; i < arg_num; i++) {
      auto blob_desc =
          GetBlobDescFromMlirTensorType(operands[operand_index].getType().cast<TensorType>());
      auto bn = arg_name + "_" + std::to_string(i);
      LOG(ERROR) << "emplace bn: " << bn;
      lbi2logical_blob_desc_.emplace(bn, std::move(blob_desc));
      operand_index += 1;
    }
  }
  for (const auto& pair : llvm::zip(operand_ids, operands)) {
    auto operand_id = std::get<0>(pair);
    auto operand = std::get<1>(pair);
    auto bn = operand_id.first + "_" + std::to_string(operand_id.second);
  }
  auto GetLogicalBlobDesc4BnInOp = [&](const std::string& bn) -> ::oneflow::BlobDesc* {
    auto it = lbi2logical_blob_desc_.find(bn);
    if (it == lbi2logical_blob_desc_.end()) { LOG(FATAL) << "fail to find bn: " << bn; }
    return it->second.get();
  };
  ::oneflow::ParallelDesc* parallel_desc = nullptr;
  CHECK_JUST(op->InferLogicalOutBlobDescs(GetLogicalBlobDesc4BnInOp, *parallel_desc));
  return failure();
}

}  // namespace oneflow

}  // namespace mlir
