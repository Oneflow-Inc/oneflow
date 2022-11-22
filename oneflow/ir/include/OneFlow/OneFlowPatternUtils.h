#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/PatternMatch.h"

namespace mlir {

namespace oneflow {

mlir::IntegerAttr GetDefaultSeed(::mlir::PatternRewriter& rewriter);

namespace rewrites {
void populateRewrites(RewritePatternSet& patterns);
}

}  // namespace oneflow

}  // namespace mlir
