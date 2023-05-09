#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_POST_TRANSFORM_CHECKS_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_POST_TRANSFORM_CHECKS_H_

#include <memory>

namespace mlir {
class Pass;

namespace oneflow {

std::unique_ptr<Pass> createOneFlowPostTransformChecksPass();

}  // namespace oneflow
}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_TRANSFORM_POST_TRANSFORM_CHECKS_H_