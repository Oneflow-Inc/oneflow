//===- TestPDLByteCode.cpp - Test PDLL functionality ----------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#include "mlir/Dialect/PDL/IR/PDL.h"
#include "mlir/Dialect/PDLInterp/IR/PDLInterp.h"
#include "mlir/Parser/Parser.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Pass/PassManager.h"
#include "mlir/Transforms/GreedyPatternRewriteDriver.h"
#include "OneFlow/OneFlowPDLLPatterns.h"

using namespace mlir;

#include "oneflow/ir/lib/OneFlow/PDLL/OneFlowPatterns.h.inc"

namespace mlir {

namespace oneflow {

void populatePDLLPatterns(RewritePatternSet& patterns) { populateGeneratedPDLLPatterns(patterns); }

}  // namespace oneflow

}  // namespace mlir
