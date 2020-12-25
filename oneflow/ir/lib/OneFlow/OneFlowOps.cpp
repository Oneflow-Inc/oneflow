//===- OneFlowOps.cpp - OneFlow dialect ops ---------------*- C++ -*-===//
//
// This file is licensed under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "OneFlow/OneFlowOps.h"
#include "OneFlow/OneFlowDialect.h"
#include "mlir/IR/OpImplementation.h"

using namespace mlir;
using namespace mlir::oneflow;

static mlir::ParseResult parseConstantOp(mlir::OpAsmParser &parser, mlir::OperationState &result) {
  mlir::DenseElementsAttr value;
  if (parser.parseOptionalAttrDict(result.attributes)
      || parser.parseAttribute(value, "value", result.attributes))
    return failure();

  result.addTypes(value.getType());
  return success();
}

static mlir::LogicalResult verify(ConstantOp op) { return mlir::success(); }

void ConstantOp::build(mlir::OpBuilder &builder, mlir::OperationState &state, double value) {
  ConstantOp::build(builder, state, RankedTensorType::get({}, builder.getF32Type()),
                    builder.getFloatAttr(builder.getF64Type(), value));
}

#define GET_OP_CLASSES
#include "OneFlow/OneFlowOps.cpp.inc"
