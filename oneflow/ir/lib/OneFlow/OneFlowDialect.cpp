#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"

using namespace mlir;
using namespace mlir::oneflow;

//===----------------------------------------------------------------------===//
// OneFlow dialect.
//===----------------------------------------------------------------------===//

void OneFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OneFlow/OneFlowOps.cpp.inc"
      >();
}
