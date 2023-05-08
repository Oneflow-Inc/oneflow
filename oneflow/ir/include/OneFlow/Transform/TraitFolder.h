#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_TRAIT_FOLDER_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_TRAIT_FOLDER_H_

#include "mlir/Pass/Pass.h"

namespace mlir {
namespace oneflow {

std::unique_ptr<mlir::Pass> createTestOneFlowTraitFolderPass();

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_TRAIT_FOLDER_H_