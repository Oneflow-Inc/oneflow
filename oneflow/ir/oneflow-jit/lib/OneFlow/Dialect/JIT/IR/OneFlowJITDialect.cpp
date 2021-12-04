#include "OneFlow/Dialect/JIT/IR/OneFlowJITDialect.h"
#include "OneFlow/Dialect/JIT/IR/OneFlowJITOps.h"
#include "OneFlow/Dialect/JIT/IR/OneFlowJITOpsDialect.cpp.inc"
namespace mlir {
namespace oneflow {
namespace jit {
void OneFlowJITDialect::initialize() {
  addOperations<
#define GET_OP_LIST
#include "OneFlow/Dialect/JIT/IR/OneFlowJITOps.cpp.inc"
      >();
}
}  // namespace jit
}  // namespace oneflow
}  // namespace mlir
