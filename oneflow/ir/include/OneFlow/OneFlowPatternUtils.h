#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

namespace oneflow {

namespace rewrites {

mlir::IntegerAttr GetDefaultSeed(::mlir::PatternRewriter& rewriter);
void populateRewrites(RewritePatternSet& patterns);

}  // namespace rewrites

}  // namespace oneflow

}  // namespace mlir
