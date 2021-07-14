#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_CONVERSION_ONEFLOWTOTOSA_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_CONVERSION_ONEFLOWTOTOSA_H_

#include "mlir/Dialect/Tosa/IR/TosaOps.h"
#include "mlir/Pass/Pass.h"

namespace mlir {

namespace oneflow {

std::unique_ptr<mlir::Pass> createLowerOneFlowToTosaPass();

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_CONVERSION_ONEFLOWTOTOSA_H_
