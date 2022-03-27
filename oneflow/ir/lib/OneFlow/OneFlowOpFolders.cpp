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
#include <memory>
#include <vector>
#include "OneFlow/OneFlowOps.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/OperationSupport.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/operator/variable_tensor_mgr.h"

namespace mlir {
namespace oneflow {

OpFoldResult VariableIrOp::fold(ArrayRef<Attribute> operands) {
  NamedAttrList attrs;
  attrs.set(valueAttrName(), valueAttr());
  attrs.set(op_nameAttrName(), op_nameAttr());
  attrs.set(device_tagAttrName(), device_tagAttr());
  attrs.set(device_nameAttrName(), device_nameAttr());
  attrs.set(scope_symbol_idAttrName(), scope_symbol_idAttr());
  attrs.set(hierarchyAttrName(), hierarchyAttr());
  attrs.set(nd_sbpAttrName(), nd_sbpAttr());
  return DictionaryAttr::get(getContext(), attrs);
}

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  ::oneflow::LazyMode::Guard guard{false};
  if (!operands.front()) { return {}; }  // Important!
  const auto attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto attrs = NamedAttrList(attr_dict);
  const auto tensor =
      support::DenseElementsAttrToTensor(attr_dict.get("value").cast<mlir::DenseElementsAttr>());
  std::vector<int64_t> shape_vec;
  for (auto& x : shape().getValue()) {
    shape_vec.emplace_back(x.cast<mlir::IntegerAttr>().getInt());
  }
  auto result =
      ::oneflow::one::functional::Reshape(
          tensor, ::oneflow::Shape(::oneflow::DimVector(shape_vec.begin(), shape_vec.end())))
          .GetPtrOrThrow();
  attrs.erase("value");
  attrs.set("value",
            support::TensorToDenseElementsAttr(result, mlir::FloatType::getF32(getContext())));
  ::oneflow::Global<::oneflow::VariableTensorMgr>::Get()
      ->Set(attr_dict.get("op_name").cast<mlir::StringAttr>().str(), result)
      .GetOrThrow();
  return attrs.getDictionary(getContext());
}

OpFoldResult ScalarAddOp::fold(ArrayRef<Attribute> operands) {
  ::oneflow::LazyMode::Guard guard{false};
  if (!operands.front()) { return {}; }  // Important!
  const auto attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto attrs = NamedAttrList(attr_dict);
  const auto tensor =
      support::DenseElementsAttrToTensor(attr_dict.get("value").cast<mlir::DenseElementsAttr>());
  std::shared_ptr<::oneflow::one::Tensor> result;
  if (has_int_operand()) {
    result = ::oneflow::one::functional::ScalarAdd(tensor, int_operand(), 1, false).GetPtrOrThrow();
  }
  if (has_float_operand()) {
    result =
        ::oneflow::one::functional::ScalarAdd(tensor, float_operand().convertToFloat(), 1, false)
            .GetPtrOrThrow();
  }
  attrs.erase("value");
  attrs.set("value",
            support::TensorToDenseElementsAttr(result, mlir::FloatType::getF32(getContext())));
  ::oneflow::Global<::oneflow::VariableTensorMgr>::Get()
      ->Set(attr_dict.get("op_name").cast<mlir::StringAttr>().str(), result)
      .GetOrThrow();
  return attrs.getDictionary(getContext());
}

OpFoldResult MultiplyOp::fold(ArrayRef<Attribute> operands) {
  ::oneflow::LazyMode::Guard guard{false};
  if (!(operands.front() && operands.back())) { return {}; }  // Important!
  auto lhs_attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto rhs_attr_dict = operands.back().cast<mlir::DictionaryAttr>();

  auto attrs = NamedAttrList(lhs_attr_dict);
  auto lhs_tensor = support::DenseElementsAttrToTensor(
      lhs_attr_dict.get("value").cast<mlir::DenseElementsAttr>());
  auto rhs_tensor = support::DenseElementsAttrToTensor(
      rhs_attr_dict.get("value").cast<mlir::DenseElementsAttr>());

  const auto result = ::oneflow::one::functional::Mul(lhs_tensor, rhs_tensor).GetPtrOrThrow();

  attrs.erase("value");
  attrs.set("value",
            support::TensorToDenseElementsAttr(result, mlir::FloatType::getF32(getContext())));
  ::oneflow::Global<::oneflow::VariableTensorMgr>::Get()
      ->Delete(rhs_attr_dict.get("op_name").cast<mlir::StringAttr>().str())
      .GetOrThrow();
  ::oneflow::Global<::oneflow::VariableTensorMgr>::Get()
      ->Set(lhs_attr_dict.get("op_name").cast<mlir::StringAttr>().str(), result)
      .GetOrThrow();
  return attrs.getDictionary(getContext());
}

OpFoldResult BroadcastDivOp::fold(ArrayRef<Attribute> operands) {
  ::oneflow::LazyMode::Guard guard{false};
  if (!(operands.front() && operands.back())) { return {}; }  // Important!
  auto lhs_attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto rhs_attr_dict = operands.back().cast<mlir::DictionaryAttr>();

  auto attrs = NamedAttrList(lhs_attr_dict);
  auto lhs_tensor = support::DenseElementsAttrToTensor(
      lhs_attr_dict.get("value").cast<mlir::DenseElementsAttr>());
  auto rhs_tensor = support::DenseElementsAttrToTensor(
      rhs_attr_dict.get("value").cast<mlir::DenseElementsAttr>());

  const auto result = ::oneflow::one::functional::Div(lhs_tensor, rhs_tensor).GetPtrOrThrow();

  attrs.erase("value");
  attrs.set("value",
            support::TensorToDenseElementsAttr(result, mlir::FloatType::getF32(getContext())));
  ::oneflow::Global<::oneflow::VariableTensorMgr>::Get()
      ->Delete(rhs_attr_dict.get("op_name").cast<mlir::StringAttr>().str())
      .GetOrThrow();
  ::oneflow::Global<::oneflow::VariableTensorMgr>::Get()
      ->Set(lhs_attr_dict.get("op_name").cast<mlir::StringAttr>().str(), result)
      .GetOrThrow();
  return attrs.getDictionary(getContext());
}

}  // namespace oneflow
}  // namespace mlir
