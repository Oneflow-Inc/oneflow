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
#include "mlir/Support/LogicalResult.h"

namespace mlir {

namespace oneflow {

namespace {

std::unique_ptr<::oneflow::BlobDesc> getBlobDescFromTensorType(TensorType tensor_type) {
  auto data_type = mlir::oneflow::support::FromMLIRTypeToOFDataType(tensor_type.getElementType());
  if (mlir::succeeded(data_type)) {
    auto shape_from_mlir = new ::oneflow::Shape(llvm::SmallVector<int64_t, 4>(
        {tensor_type.getShape().begin(), tensor_type.getShape().end()}));
    return std::make_unique<::oneflow::BlobDesc>(*shape_from_mlir, data_type.getValue());
  }
  tensor_type.dump();
  LOG(FATAL) << "fail to get BlobDesc from TensorType";
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

size_t getResultSize(DictionaryAttr attributes) {
  const StringRef attr_name = OpTrait::AttrSizedResultSegments<void>::getResultSegmentSizeAttr();
  const DenseIntElementsAttr& size_attr =
      attributes.get(attr_name).dyn_cast_or_null<DenseIntElementsAttr>();
  CHECK(size_attr) << "Attr " << attr_name.str() << " is not found or not DenseIntElementsAttr";
  auto size = 0;
  for (auto s : size_attr.getValues<int32_t>()) { size += s; }
  return size;
}

::mlir::LogicalResult inferReturnTypesWithOpTypeName(
    llvm::StringRef op_type_name, ::mlir::MLIRContext* context, ::mlir::ValueRange operands,
    ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  ::oneflow::OperatorConf op_conf{};
  CHECK(ConvertUserOp(op_type_name, op_conf, operands, attributes).succeeded());
  std::unordered_map<std::string, std::unique_ptr<::oneflow::BlobDesc>> lbi2logical_blob_desc_;
  auto operand_ids =
      user_op::ArgIds<OpTrait::AttrSizedOperandSegments>(op_type_name, operands.size(), attributes);
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
  auto result_ids = user_op::ArgIds<OpTrait::AttrSizedResultSegments>(
      op_type_name, getResultSize(attributes), attributes);
  for (const auto& result_id : result_ids) {
    const auto& arg_name = result_id.first;
    const auto& arg_id = result_id.second;
    const auto bn = ::oneflow::GenRepeatedBn(arg_name, arg_id);
    auto blob_desc = std::make_unique<::oneflow::BlobDesc>(::oneflow::kInvalidDataType);
    lbi2logical_blob_desc_.emplace(bn, std::move(blob_desc));
    (*op_conf.mutable_user_conf()->mutable_output())[arg_name].add_s(
        ::oneflow::GenLogicalBlobName(op_conf.name(), bn));
  }
  auto op = CHECK_JUST(ConstructOp(op_conf, user_op::getDeviceTypeFromAttrDictionary(attributes)));
  auto GetLogicalBlobDesc4BnInOp = [&](const std::string& bn) -> ::oneflow::BlobDesc* {
    auto it = lbi2logical_blob_desc_.find(bn);
    if (it == lbi2logical_blob_desc_.end()) {
      LOG(FATAL) << "fail to find blob name in op: " << bn;
    }
    return it->second.get();
  };
  ::oneflow::ParallelConf parallel_conf = user_op::getParallelConfFromAttrDictionary(attributes);
  ::oneflow::ParallelDesc parallel_desc{parallel_conf};
  op->FillOpParallelDesc(parallel_desc);
  CHECK_JUST(op->InferLogicalOutBlobDescs(GetLogicalBlobDesc4BnInOp, parallel_desc));
  for (const auto& result_id : result_ids) {
    const auto& arg_name = result_id.first;
    const auto& arg_id = result_id.second;
    const auto bn = ::oneflow::GenRepeatedBn(arg_name, arg_id);
    const auto* desc = lbi2logical_blob_desc_.at(bn).get();
    if (auto t = getTensorTypeFromBlobDesc(context, desc)) { inferredReturnTypes.push_back(t); }
  }
  return success();
}

}  // namespace

::mlir::LogicalResult NormalizationAddReluOp::refineReturnTypes(
    ::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  return success();
}

::mlir::LogicalResult NormalizationAddReluOp::inferReturnTypes(
    ::mlir::MLIRContext* context, ::llvm::Optional<::mlir::Location> location,
    ::mlir::ValueRange operands, ::mlir::DictionaryAttr attributes, ::mlir::RegionRange regions,
    ::llvm::SmallVectorImpl<::mlir::Type>& inferredReturnTypes) {
  return inferReturnTypesWithOpTypeName("normalization_add_relu", context, operands, attributes,
                                        regions, inferredReturnTypes);
}

}  // namespace oneflow

}  // namespace mlir
