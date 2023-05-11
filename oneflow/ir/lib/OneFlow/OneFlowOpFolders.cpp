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
#include <functional>
#include <memory>
#include <vector>
#include "OneFlow/OneFlowOpTraits.h"
#include "OneFlow/OneFlowOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OpDefinition.h"
#include "mlir/IR/OperationSupport.h"
#include "mlir/IR/Value.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/framework/variable_tensor_mgr.h"

namespace mlir {
namespace oneflow {
namespace {

namespace functional = ::oneflow::one::functional;
using TensorPtr = std::shared_ptr<::oneflow::one::Tensor>;
using MaybeTensor = ::oneflow::Maybe<::oneflow::one::Tensor>;

StringAttr GenNewVariableOpName(MLIRContext* ctx, const std::string& key = "") {
  if (key == "") { return StringAttr::get(ctx, "variable_" + ::oneflow::NewUniqueId()); }
  return StringAttr::get(ctx, "variable_" + key + "_" + ::oneflow::NewUniqueId());
}

bool MLIRDataTypesAreSame(const std::vector<DataType>& data_types) {
  if (data_types.empty() || data_types.size() == 1) { return true; }
  bool result = true;
  const auto first_data_type = data_types[0];
  for (size_t i = 1; i < data_types.size(); ++i) { result &= (first_data_type == data_types[i]); }
  return result;
}

bool DictionaryAttrsHaveSameDataType(const std::vector<mlir::DictionaryAttr>& attrs) {
  std::vector<DataType> data_types;
  for (const auto& attr : attrs) {
    data_types.push_back(attr.get(OpTrait::TensorSource<void>::getDataTypeAttrName())
                             .cast<DataTypeAttr>()
                             .getValue());
  }
  return MLIRDataTypesAreSame(data_types);
}

OpFoldResult UnaryFold(MLIRContext* ctx, ArrayRef<Attribute> operands,
                       const std::function<MaybeTensor(const TensorPtr&)>& f) {
  ::oneflow::LazyMode::Guard guard{false};
  if (!operands.front()) { return {}; }  // Important!

  const auto attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto attrs = NamedAttrList(attr_dict);
  const auto tensor = support::DenseElementsAttrToTensor(
      attr_dict.get("value"), attr_dict.get(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr()),
      attr_dict.get(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr()));
  const auto result = f(tensor).GetPtrOrThrow();
  attrs.set("value", support::TensorToDenseElementsAttr(result, ctx));
  attrs.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(), GenNewVariableOpName(ctx));
  attrs.set(OpTrait::TensorSource<void>::getDataTypeAttrName(),
            attr_dict.get(OpTrait::TensorSource<void>::getDataTypeAttrName()));

  return attrs.getDictionary(ctx);
}

OpFoldResult BinaryFold(MLIRContext* ctx, ArrayRef<Attribute> operands,
                        const std::function<MaybeTensor(const TensorPtr&, const TensorPtr&)>& f) {
  ::oneflow::LazyMode::Guard guard{false};
  if (!(operands.front() && operands.back())) { return {}; }  // Important!
  auto lhs_attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto rhs_attr_dict = operands.back().cast<mlir::DictionaryAttr>();
  if (!DictionaryAttrsHaveSameDataType({lhs_attr_dict, rhs_attr_dict})) {
    llvm::errs()
        << "Input tensors should have same data type in binary operation of constant folding."
        << "\n";
    return nullptr;
  }

  auto attrs = NamedAttrList(lhs_attr_dict);
  const auto lhs_tensor = support::DenseElementsAttrToTensor(
      lhs_attr_dict.get("value"),
      lhs_attr_dict.get(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr()),
      lhs_attr_dict.get(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr()));
  const auto rhs_tensor = support::DenseElementsAttrToTensor(
      rhs_attr_dict.get("value"),
      rhs_attr_dict.get(OpTrait::IsOpConfCompatible<void>::getDeviceTagAttr()),
      rhs_attr_dict.get(OpTrait::IsOpConfCompatible<void>::getDeviceNameAttr()));

  const auto result = f(lhs_tensor, rhs_tensor).GetPtrOrThrow();

  attrs.set("value", support::TensorToDenseElementsAttr(result, ctx));
  attrs.set(OpTrait::IsOpConfCompatible<void>::getOpNameAttr(), GenNewVariableOpName(ctx));
  attrs.set(OpTrait::TensorSource<void>::getDataTypeAttrName(),
            lhs_attr_dict.get(OpTrait::TensorSource<void>::getDataTypeAttrName()));

  return attrs.getDictionary(ctx);
}

}  // namespace

OpFoldResult FrozenVariableOp::fold(FoldAdaptor adaptor) {
  NamedAttrList attrs;
  attrs.set(getValueAttrName(), getValueAttr());
  attrs.set(getOpNameAttrName(), getOpNameAttr());
  attrs.set(getDataTypeAttrName(), getDataTypeAttr());
  attrs.set(getDeviceTagAttrName(), getDeviceTagAttr());
  attrs.set(getDeviceNameAttrName(), getDeviceNameAttr());
  attrs.set(getScopeSymbolIdAttrName(), getScopeSymbolIdAttr());
  attrs.set(getHierarchyAttrName(), getHierarchyAttr());
  attrs.set(getNdSbpAttrName(), getNdSbpAttr());
  return DictionaryAttr::get(getContext(), attrs);
}

OpFoldResult TransposeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return UnaryFold(getContext(), operands, [this](const auto& tensor) {
    std::vector<int32_t> perm_;
    for (auto& x : getPerm().getValue()) { perm_.emplace_back(x.cast<IntegerAttr>().getSInt()); }
    return functional::Transpose(tensor, perm_);
  });
}

OpFoldResult ReshapeOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return UnaryFold(getContext(), operands, [this](const auto& tensor) {
    std::vector<int64_t> shape_vec;
    for (auto& x : getShape().getValue()) {
      shape_vec.emplace_back(x.cast<mlir::IntegerAttr>().getValue().getSExtValue());
    }
    return functional::Reshape(
        tensor, ::oneflow::Shape(::oneflow::DimVector(shape_vec.begin(), shape_vec.end())));
  });
}

OpFoldResult ScalarAddOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return UnaryFold(getContext(), operands, [this](const auto& tensor) -> MaybeTensor {
    if (getHasIntOperand()) { return functional::ScalarAdd(tensor, getIntOperand(), 1, false); }
    if (getHasFloatOperand()) {
      return functional::ScalarAdd(tensor, getFloatOperand().convertToDouble(), 1, false);
    }
    emitError("Scalar op must has a int operand or a float operand.");
    return TensorPtr();
  });
}

OpFoldResult SqrtOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return UnaryFold(getContext(), operands, functional::Sqrt);
}

OpFoldResult BroadcastMulOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return BinaryFold(getContext(), operands, functional::Mul);
}

OpFoldResult BroadcastDivOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return BinaryFold(getContext(), operands, functional::Div);
}

OpFoldResult BroadcastSubOp::fold(FoldAdaptor adaptor) {
  auto operands = adaptor.getOperands();
  return BinaryFold(getContext(), operands, [](const auto& lhs, const auto& rhs) -> MaybeTensor {
    return functional::Sub(lhs, rhs, /*alpha=*/1.0, false);
  });
}

}  // namespace oneflow
}  // namespace mlir
