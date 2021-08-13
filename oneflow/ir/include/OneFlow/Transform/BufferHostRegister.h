#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_BUFFERHOSTREGISTER_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_BUFFERHOSTREGISTER_H_

#include "mlir/Pass/Pass.h"

namespace mlir {

namespace oneflow {

std::unique_ptr<mlir::Pass> createBufferHostRegisterPass();

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_BUFFERHOSTREGISTER_H_
