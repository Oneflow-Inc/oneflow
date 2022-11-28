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
#include "OneFlow/OneFlowDataTypeConversion.h"

namespace mlir {

namespace oneflow {

namespace {

std::unique_ptr<::oneflow::BlobDesc> getBlobDescFromTensorType(TensorType tensor_type) {
  auto data_type = mlir::oneflow::support::FromMLIRTypeToOFDataType(tensor_type.getElementType());
  if (mlir::succeeded(data_type)) {
    auto shape_from_mlir = new ::oneflow::Shape(llvm::SmallVector<int64_t, 4>(
        {tensor_type.getShape().begin(), tensor_type.getShape().end()}));
    return std::make_unique<::oneflow::BlobDesc>(*shape_from_mlir, data_type.getValue());
  } else {
    tensor_type.dump();
    LOG(FATAL) << "fail to get BlobDesc from TensorType";
  }
}

Type getTensorTypeFromBlobDesc(MLIRContext* context, const ::oneflow::BlobDesc* blob_desc) {
  if (auto type = getTypeFromOneFlowDataType(context, blob_desc->data_type())) {
    return RankedTensorType::get(
        llvm::SmallVector<int64_t, 4>(
            {blob_desc->shape().dim_vec().begin(), blob_desc->shape().dim_vec().end()}),
        type);
  } else {
    return Type{};
  }
}

static auto MagicalOpName = "INFER_MAGICAL";
LogicalResult ConvertUserOp(llvm::StringRef op_type_name, ::oneflow::OperatorConf& op_conf,
                            ValueRange operands, DictionaryAttr attributes) {
  oneflow::ConfOpAdaptor conf_op_adaptor(operands, attributes);
  op_conf.set_name(MagicalOpName);
  CHECK(
      user_op::ConvertUserOpInputs(op_type_name, operands, attributes, op_conf.mutable_user_conf())
          .succeeded());
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
  std::unordered_map<std::string, std::unique_ptr<::oneflow::BlobDesc>> lbi2logical_blob_desc_;
  auto operand_ids =
      user_op::ArgIds<OpTrait::AttrSizedOperandSegments>(op_type_name, operands, attributes);
  auto operand_index = 0;
  for (const auto& idOperand : llvm::zip(operand_ids, operands)) {
    const auto& arg_name = std::get<0>(idOperand).first;
    const auto& arg_id = std::get<0>(idOperand).second;
    const auto operand = std::get<1>(idOperand);
    auto blob_desc = getBlobDescFromTensorType(operand.getType().cast<TensorType>());
    auto bn = ::oneflow::GenRepeatedBn(arg_name, arg_id);
    lbi2logical_blob_desc_.emplace(bn, std::move(blob_desc));
    operand_index += 1;
  }
  static auto MAX_OUTPUT_NUM_PER_BN = 1;
  for (const auto& arg_name : support::GetOutputKeys(op_type_name)) {
    for (size_t arg_id = 0; arg_id < MAX_OUTPUT_NUM_PER_BN; arg_id++) {
      auto blob_desc = std::make_unique<::oneflow::BlobDesc>(::oneflow::kInvalidDataType);
      auto bn = ::oneflow::GenRepeatedBn(arg_name, arg_id);
      lbi2logical_blob_desc_.emplace(bn, std::move(blob_desc));
      (*op_conf.mutable_user_conf()->mutable_output())[arg_name].add_s(
          ::oneflow::GenLogicalBlobName(op_conf.name(), bn));
    }
  }
  auto op = CHECK_JUST(ConstructOp(op_conf, user_op::generateDeviceName(attributes)));
  auto GetLogicalBlobDesc4BnInOp = [&](const std::string& bn) -> ::oneflow::BlobDesc* {
    auto it = lbi2logical_blob_desc_.find(bn);
    if (it == lbi2logical_blob_desc_.end()) { LOG(FATAL) << "fail to find bn: " << bn; }
    return it->second.get();
  };
  ::oneflow::ParallelConf parallel_conf = user_op::generateParallelConf(attributes);
  ::oneflow::ParallelDesc parallel_desc{parallel_conf};
  op->FillOpParallelDesc(parallel_desc);
  CHECK_JUST(op->InferLogicalOutBlobDescs(GetLogicalBlobDesc4BnInOp, parallel_desc));
  for (const auto& arg_name : support::GetOutputKeys(op_type_name)) {
    for (size_t arg_id = 0; arg_id < MAX_OUTPUT_NUM_PER_BN; arg_id++) {
      const auto bn = ::oneflow::GenRepeatedBn(arg_name, arg_id);
      const auto* desc = lbi2logical_blob_desc_.at(bn).get();
      if (auto t = getTensorTypeFromBlobDesc(context, desc)) { inferredReturnTypes.push_back(t); }
    }
  }
  return success();
}

}  // namespace oneflow

}  // namespace mlir
