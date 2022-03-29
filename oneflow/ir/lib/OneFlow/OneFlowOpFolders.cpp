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
#include "mlir/IR/OperationSupport.h"
#include "oneflow/core/common/shape_vec.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/functional/functional_api.yaml.h"
#include "oneflow/core/job/lazy_mode.h"
#include "oneflow/core/operator/variable_tensor_mgr.h"

namespace mlir {
namespace oneflow {
namespace {

class FolderGuard final {
 public:
  explicit FolderGuard(MLIRContext* context) : context_(context) {}
  const StringAttr GenNewVariableOpName(const std::string& key = "") const {
    if (key == "") { return StringAttr::get(context_, "variable_" + ::oneflow::NewUniqueId()); }
    return StringAttr::get(context_, "variable_" + key + "_" + ::oneflow::NewUniqueId());
  }

 private:
  MLIRContext* context_ = nullptr;
  ::oneflow::LazyMode::Guard guard{false};
};

using TensorPtr = std::shared_ptr<::oneflow::one::Tensor>;

OpFoldResult UnaryFold(MLIRContext* ctx, ArrayRef<Attribute> operands,
                       const std::function<TensorPtr(const TensorPtr&)>& f) {
  const FolderGuard g(ctx);
  if (!operands.front()) { return {}; }  // Important!

  const auto attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto attrs = NamedAttrList(attr_dict);
  const auto tensor =
      support::DenseElementsAttrToTensor(attr_dict.get("value").cast<mlir::DenseElementsAttr>());
  const auto result = f(tensor);
  attrs.set("value", support::TensorToDenseElementsAttr(result, mlir::FloatType::getF32(ctx)));
  attrs.set("op_name", g.GenNewVariableOpName());

  return attrs.getDictionary(ctx);
}

OpFoldResult BinaryFold(MLIRContext* ctx, ArrayRef<Attribute> operands,
                        const std::function<TensorPtr(const TensorPtr&, const TensorPtr&)>& f) {
  const FolderGuard g(ctx);
  if (!(operands.front() && operands.back())) { return {}; }  // Important!
  auto lhs_attr_dict = operands.front().cast<mlir::DictionaryAttr>();
  auto rhs_attr_dict = operands.back().cast<mlir::DictionaryAttr>();

  auto attrs = NamedAttrList(lhs_attr_dict);
  const auto lhs_tensor = support::DenseElementsAttrToTensor(
      lhs_attr_dict.get("value").cast<mlir::DenseElementsAttr>());
  const auto rhs_tensor = support::DenseElementsAttrToTensor(
      rhs_attr_dict.get("value").cast<mlir::DenseElementsAttr>());

  const auto result = f(lhs_tensor, rhs_tensor);

  attrs.set("value", support::TensorToDenseElementsAttr(result, mlir::FloatType::getF32(ctx)));
  attrs.set("op_name", g.GenNewVariableOpName());

  return attrs.getDictionary(ctx);
}

}  // namespace

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
  return UnaryFold(getContext(), operands, [this](const auto& tensor) {
    std::vector<int64_t> shape_vec;
    for (auto& x : shape().getValue()) {
      shape_vec.emplace_back(x.cast<mlir::IntegerAttr>().getInt());
    }
    return ::oneflow::one::functional::Reshape(
               tensor, ::oneflow::Shape(::oneflow::DimVector(shape_vec.begin(), shape_vec.end())))
        .GetPtrOrThrow();
  });
}

OpFoldResult ScalarAddOp::fold(ArrayRef<Attribute> operands) {
  return UnaryFold(getContext(), operands, [this](const auto& tensor) -> TensorPtr {
    if (has_int_operand()) {
      return ::oneflow::one::functional::ScalarAdd(tensor, int_operand(), 1, false).GetPtrOrThrow();
    }
    if (has_float_operand()) {
      return ::oneflow::one::functional::ScalarAdd(tensor, float_operand().convertToFloat(), 1,
                                                   false)
          .GetPtrOrThrow();
    }
    return nullptr;
  });
}

OpFoldResult SqrtOp::fold(ArrayRef<Attribute> operands) {
  return UnaryFold(getContext(), operands, [](const auto& tensor) -> TensorPtr {
    return ::oneflow::one::functional::Sqrt(tensor).GetPtrOrThrow();
  });
}

OpFoldResult MultiplyOp::fold(ArrayRef<Attribute> operands) {
  return BinaryFold(getContext(), operands, [](const auto& lhs, const auto& rhs) -> TensorPtr {
    return ::oneflow::one::functional::Mul(lhs, rhs).GetPtrOrThrow();
  });
}

OpFoldResult BroadcastDivOp::fold(ArrayRef<Attribute> operands) {
  return BinaryFold(getContext(), operands, [](const auto& lhs, const auto& rhs) -> TensorPtr {
    return ::oneflow::one::functional::Div(lhs, rhs).GetPtrOrThrow();
  });
}

OpFoldResult BroadcastSubOp::fold(ArrayRef<Attribute> operands) {
  return BinaryFold(getContext(), operands, [](const auto& lhs, const auto& rhs) -> TensorPtr {
    return ::oneflow::one::functional::Sub(lhs, rhs, false).GetPtrOrThrow();
  });
}

}  // namespace oneflow
}  // namespace mlir
