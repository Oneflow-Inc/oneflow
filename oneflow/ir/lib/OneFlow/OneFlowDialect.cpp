#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"

using namespace mlir;
using namespace mlir::oneflow;

void OneFlowDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OneFlow/OneFlowOps.cpp.inc"
      >();
}
