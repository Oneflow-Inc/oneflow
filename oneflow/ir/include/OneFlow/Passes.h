#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_PASSES_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_PASSES_H_

#include "mlir/Dialect/MemRef/IR/MemRef.h"
#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Dialect/StandardOps/IR/Ops.h"
#include "OneFlow/Conversion/OneFlowToTosa.h"

namespace mlir {

#define GEN_PASS_CLASSES
#define GEN_PASS_REGISTRATION
#include "OneFlow/OneFlowPasses.h.inc"

namespace oneflow {

LogicalResult Lower(mlir::MLIRContext* context, OwningModuleRef& module);
void populateFuserPasses(::mlir::RewritePatternSet& patterns);

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_PASSES_H_
