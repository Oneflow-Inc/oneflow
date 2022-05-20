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

OpFoldResult UnaryFold(MLIRContext* ctx, ArrayRef<Attribute> operands,
                       const std::function<MaybeTensor(const TensorPtr&)>& f) {
  ::oneflow::LazyMode::Guard guard{false};
  if (!operands.front()) { return {}; }  // Important!

  const auto attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto attrs = NamedAttrList(attr_dict);
  const auto tensor = support::DenseElementsAttrToTensor(
      attr_dict.get("value"), attr_dict.get("device_tag"), attr_dict.get("device_name"));
  const auto result = f(tensor).GetPtrOrThrow();
  attrs.set("value", support::TensorToDenseElementsAttr(result, ctx));
  attrs.set("op_name", GenNewVariableOpName(ctx));

  return attrs.getDictionary(ctx);
}

OpFoldResult BinaryFold(MLIRContext* ctx, ArrayRef<Attribute> operands,
                        const std::function<MaybeTensor(const TensorPtr&, const TensorPtr&)>& f) {
  ::oneflow::LazyMode::Guard guard{false};
  if (!(operands.front() && operands.back())) { return {}; }  // Important!
  auto lhs_attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto rhs_attr_dict = operands.back().cast<mlir::DictionaryAttr>();

  auto attrs = NamedAttrList(lhs_attr_dict);
  const auto lhs_tensor = support::DenseElementsAttrToTensor(lhs_attr_dict.get("value"),
                                                             lhs_attr_dict.get("device_tag"),
                                                             lhs_attr_dict.get("device_name"));
  const auto rhs_tensor = support::DenseElementsAttrToTensor(rhs_attr_dict.get("value"),
                                                             rhs_attr_dict.get("device_tag"),
                                                             rhs_attr_dict.get("device_name"));

  const auto result = f(lhs_tensor, rhs_tensor).GetPtrOrThrow();

  attrs.set("value", support::TensorToDenseElementsAttr(result, ctx));
  attrs.set("op_name", GenNewVariableOpName(ctx));

  return attrs.getDictionary(ctx);
}

}  // namespace

OpFoldResult FrozenVariableOp::fold(ArrayRef<Attribute> operands) {
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

OpFoldResult TransposeOp::fold(ArrayRef<Attribute> operands) {
  return UnaryFold(getContext(), operands, [this](const auto& tensor) {
    std::vector<int32_t> perm_;
    for (auto& x : perm().getValue()) { perm_.emplace_back(x.cast<IntegerAttr>().getSInt()); }
    return functional::Transpose(tensor, perm_);
  });
}

OpFoldResult ReshapeOp::fold(ArrayRef<Attribute> operands) {
  return UnaryFold(getContext(), operands, [this](const auto& tensor) {
    std::vector<int64_t> shape_vec;
    for (auto& x : shape().getValue()) {
      shape_vec.emplace_back(x.cast<mlir::IntegerAttr>().getValue().getSExtValue());
    }
    return functional::Reshape(
        tensor, ::oneflow::Shape(::oneflow::DimVector(shape_vec.begin(), shape_vec.end())));
  });
}

OpFoldResult ScalarAddOp::fold(ArrayRef<Attribute> operands) {
  return UnaryFold(getContext(), operands, [this](const auto& tensor) -> MaybeTensor {
    if (has_int_operand()) { return functional::ScalarAdd(tensor, int_operand(), 1, false); }
    if (has_float_operand()) {
      return functional::ScalarAdd(tensor, float_operand().convertToFloat(), 1, false);
    }
    emitError("Scalar op must has a int operand or a float operand.");
    return TensorPtr();
  });
}

OpFoldResult SqrtOp::fold(ArrayRef<Attribute> operands) {
  return UnaryFold(getContext(), operands, functional::Sqrt);
}

OpFoldResult MultiplyOp::fold(ArrayRef<Attribute> operands) {
  return BinaryFold(getContext(), operands, functional::Mul);
}

OpFoldResult BroadcastDivOp::fold(ArrayRef<Attribute> operands) {
  return BinaryFold(getContext(), operands, functional::Div);
}

OpFoldResult BroadcastSubOp::fold(ArrayRef<Attribute> operands) {
  return BinaryFold(getContext(), operands, [](const auto& lhs, const auto& rhs) -> MaybeTensor {
    return functional::Sub(lhs, rhs, /*alpha=*/1.0, false);
  });
}

}  // namespace oneflow
}  // namespace mlir
