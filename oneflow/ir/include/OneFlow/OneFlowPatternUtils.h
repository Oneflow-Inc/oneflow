#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

namespace oneflow {

mlir::IntegerAttr GetDefaultSeed(::mlir::PatternRewriter& rewriter);

}  // namespace oneflow

}  // namespace mlir
