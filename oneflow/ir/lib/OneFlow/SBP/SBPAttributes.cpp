#include "OneFlow/SBP/SBPDialect.h"
#include "OneFlow/SBP/SBPAttributes.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/DialectImplementation.h"

#define GET_ATTRDEF_CLASSES
#include "OneFlow/SBPAttributes.cpp.inc"

namespace mlir {

namespace sbp {

void SBPDialect::registerAttributes() {
  addAttributes<
#define GET_ATTRDEF_LIST
#include "OneFlow/SBPAttributes.cpp.inc"
      >();
}

}  // namespace sbp

}  // namespace mlir
