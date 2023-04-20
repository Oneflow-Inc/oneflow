#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_ONEFLOW_MEMPOOL_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_ONEFLOW_MEMPOOL_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace oneflow {

namespace codegen {
namespace mempool {

inline const std::string MEMPOOL_ATTR_NAME = "memory pool";

}  // namespace mempool
}  // namespace codegen

std::unique_ptr<mlir::Pass> createFoldAllocToSubviewPass();
std::unique_ptr<mlir::Pass> createInsertOneFlowMemPoolPass();

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_ONEFLOW_MEMPOOL_H_