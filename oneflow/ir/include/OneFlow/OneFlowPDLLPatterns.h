#ifndef ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWPDLLPATTERNS_H_
#define ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWPDLLPATTERNS_H_
#include "mlir/IR/PatternMatch.h"

namespace mlir {

namespace oneflow {

void populatePDLLPasses(RewritePatternSet& patterns);

}  // namespace oneflow

}  // namespace mlir

#endif  // ONEFLOW_IR_INCLUDE_ONEFLOW_ONEFLOWPDLLPATTERNS_H_
