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
LogicalResult ConvertUserOp(::oneflow::OperatorConf& op_conf, ValueRange operands,
                            DictionaryAttr attributes) {
  oneflow::ConfOpAdaptor conf_op_adaptor(operands, attributes);
  std::string op_name = MagicalOpName;
  auto& user_conf = *op_conf.mutable_user_conf();
  // if (!succeeded(ConvertUserOpInputs(op, op_name, user_conf))) {
  //   op->emitError("fail to convert user op inputs");
  //   return failure();
  // }
  // if (!succeeded(ConvertUserOpOutputs(op, op_name, user_conf))) {
  //   op->emitError("fail to convert user op outputs");
  //   return failure();
  // }
  std::string op_type_name =
      attributes.get(OpTrait::IsAlternative<void>::getOpTypeNameAttr()).cast<StringAttr>().str();
  if (!succeeded(user_op::ConvertUserOpAttributes(op_type_name, attributes, op_conf))) {
    return failure();
  }
  return success();
}

}  // namespace

::mlir::LogicalResult NormalizationAddReluOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  TODO();
  ::oneflow::OperatorConf op_conf{};
  auto op = CHECK_JUST(ConstructOp(op_conf));
  ::oneflow::HashMap<std::string, std::unique_ptr<::oneflow::BlobDesc>> lbi2logical_blob_desc_;
  std::unordered_map<std::string, mlir::Value> operand_mapping_;
  auto GetLogicalBlobDesc4BnInOp = [&](const std::string& bn) -> ::oneflow::BlobDesc* {
    if (lbi2logical_blob_desc_.find(bn) == lbi2logical_blob_desc_.end()) {
      auto operand_it = operand_mapping_.find(bn);
      if (operand_it == operand_mapping_.end()) {
        auto blob_desc = std::make_unique<::oneflow::BlobDesc>(::oneflow::kInvalidDataType);
        CHECK(lbi2logical_blob_desc_.emplace(bn, std::move(blob_desc)).second);
      } else {
        auto found = GetBlobDescFromMlirTensorType(operand_it->second.getType().cast<TensorType>());
        CHECK(lbi2logical_blob_desc_.emplace(bn, std::move(found)).second);
      }
    }
    return lbi2logical_blob_desc_.at(bn).get();
  };
  ::oneflow::ParallelDesc* parallel_desc = nullptr;
  CHECK_JUST(op->InferLogicalOutBlobDescs(GetLogicalBlobDesc4BnInOp, *parallel_desc));
  return failure();
}

}  // namespace oneflow

}  // namespace mlir
