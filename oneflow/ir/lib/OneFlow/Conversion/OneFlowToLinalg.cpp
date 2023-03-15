

#include "OneFlow/Passes.h"
namespace mlir {

namespace oneflow {

namespace {

struct OneFlowLoweringToLinalgPass
    : public LowerOneFlowToLinalgPassBase<OneFlowLoweringToLinalgPass> {
  void runOnOperation() {}
};

}  // namespace

std::unique_ptr<Pass> createLowerOneFlowToLinalgPass() {
  return std::make_unique<OneFlowLoweringToLinalgPass>();
}

}  // namespace oneflow
}  // namespace mlir
