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
#include <iostream>
#include <string>
#include "OneFlow/OneFlowDialect.h"
#include "llvm/ADT/STLExtras.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/OpImplementation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
using namespace mlir::oneflow;

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser& parser, mlir::OperationState& result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes)
      || parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

static mlir::LogicalResult verify(ConstantOp op) { return mlir::success(); }

template<typename OpType>
LogicalResult TrimRedundantCtrl(OpType& op, PatternRewriter& rewriter) {
  const int32_t num_ctrl_outputs =
      *(op.result_segment_sizes().template getValues<uint32_t>()).end();
  if (op.ctrl_output() && op.ctrl_output().use_empty() && num_ctrl_outputs != 0) {
    const int32_t num_data_outputs =
        *(op.result_segment_sizes().template getValues<uint32_t>()).begin();
    NamedAttrList attributes(op->getAttrDictionary());
    attributes.erase("result_segment_sizes");
    attributes.append("result_segment_sizes", rewriter.getI32VectorAttr({num_data_outputs, 0}));
    if (auto created =
            rewriter.create<OpType>(op->getLoc(), op.getODSResults(0 /* data out */).getTypes(),
                                    op->getOperands(), attributes)) {
      for (auto out : op.data_output()) {
        out.replaceAllUsesWith(created->getResult(out.getResultNumber()));
      }
      op->erase();
      return success();
    }
  }
  return failure();
}

struct ConcreteUserOps : public mlir::OpRewritePattern<oneflow::UserOp> {
  ConcreteUserOps(mlir::MLIRContext* context)
      : OpRewritePattern<oneflow::UserOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(oneflow::UserOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    auto op_type_name = op->getAttrOfType<StringAttr>("op_type_name").getValue();
    op.getODSResults(0);
    if (succeeded(TrimRedundantCtrl(op, rewriter))) {
      return success();
    } else if (/* convert opaque elementwise user op to a concrete op */ op_type_name.equals("abs")
               || op_type_name.equals("ceil") || op_type_name.equals("floor")
               || op_type_name.equals("relu") || op_type_name.equals("rint")
               || op_type_name.equals("round") || op_type_name.equals("sign")
               || op_type_name.equals("negative") || op_type_name.equals("reciprocal")) {
      NamedAttrList attributes(op->getAttrDictionary());
      attributes.erase("operand_segment_sizes");
      attributes.erase("result_segment_sizes");
      auto unknownLoc = UnknownLoc::get(rewriter.getContext());
      OperationState state(unknownLoc, "oneflow." + op_type_name.str());
      state.addAttributes(attributes);
      state.addOperands(op->getOperands());
      assert(op.data_input().size() == 1);
      assert(op.data_output().size() == 1);
      state.addTypes(op.getODSResults(0 /* data out */).getTypes());
      if (auto elementwise = rewriter.createOperation(state)) {
        op.data_output().front().replaceAllUsesWith(elementwise->getResult(0));
        op->erase();
        return success();
      }
    }
    return failure();
  }
};

void UserOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                         ::mlir::MLIRContext* context) {
  results.insert<ConcreteUserOps>(context);
}

struct ConcreteSystemOps : public mlir::OpRewritePattern<oneflow::SystemOp> {
  ConcreteSystemOps(mlir::MLIRContext* context)
      : OpRewritePattern<oneflow::SystemOp>(context, /*benefit=*/1) {}
  mlir::LogicalResult matchAndRewrite(oneflow::SystemOp op,
                                      mlir::PatternRewriter& rewriter) const override {
    return TrimRedundantCtrl(op, rewriter);
  }
};

void SystemOp::getCanonicalizationPatterns(::mlir::RewritePatternSet& results,
                                           ::mlir::MLIRContext* context) {
  results.insert<ConcreteSystemOps>(context);
}

bool HaveIdenticalPlacement(mlir::Operation* op, mlir::Operation* argument_op) {
  // TODO: use things like adaptor to make it more type safe
  return op->hasAttr("device_tag") && argument_op->hasAttr("device_tag")
         && (op->getAttrOfType<StringAttr>("device_tag")
             == argument_op->getAttrOfType<StringAttr>("device_tag"))
         && op->hasAttr("device_name") && argument_op->hasAttr("device_name")
         && (op->getAttrOfType<ArrayAttr>("device_name")
             == argument_op->getAttrOfType<ArrayAttr>("device_name"));
}

OpFoldResult OpTrait::impl::foldIdempotentOfIdenticalPlacement(Operation* op) {
  auto* argument_op = op->getOperand(0).getDefiningOp();
  if (argument_op && op->getName() == argument_op->getName()
      && HaveIdenticalPlacement(op, argument_op)) {
    return op->getOperand(0);
  }
  return {};
}

OpFoldResult OpTrait::impl::foldInvolutionOfIdenticalPlacement(Operation* op) {
  auto* argument_op = op->getOperand(0).getDefiningOp();
  if (argument_op && op->getName() == argument_op->getName()
      && HaveIdenticalPlacement(op, argument_op)) {
    return argument_op->getOperand(0);
  }
  return {};
}

#include "OneFlow/OneFlowEnums.cpp.inc"

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.cpp.inc"
