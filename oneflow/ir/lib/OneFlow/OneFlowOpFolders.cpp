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
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OperationSupport.h"
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

}  // namespace oneflow
}  // namespace mlir
