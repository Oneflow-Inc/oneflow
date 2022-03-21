#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_CONSTANTFOLDING_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_CONSTANTFOLDING_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace oneflow {

std::unique_ptr<mlir::Pass> createConstantFoldingPass();

}  // namespace oneflow

}  // namespace mlir

#endif // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_CONSTANTFOLDING_H_
