#include "OneFlow/OneFlowDialect.h"
#include "oneflow/ir/include/OneFlow/OneFlowAttributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "OneFlow/OneFlowAttributes.cpp.inc"

namespace mlir {

namespace oneflow {

void OneFlowDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "OneFlow/OneFlowAttributes.cpp.inc"
      >();
}

}  // namespace oneflow

}  // namespace mlir
