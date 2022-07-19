#include "OneFlow/SBP/SBPDialect.h"
#include "OneFlow/SBP/SBPAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;
// [[#sbp.b]]

LogicalResult parseSBP(AsmParser& parser, FailureOr<ArrayAttr>& args) {
  if (failed(parser.parseLSquare())) { return failure(); }
  if (succeeded(parser.parseOptionalRSquare())) { return success(); }
  llvm::SmallVector<Attribute> res;
  if (parser.parseCommaSeparatedList([&]() { return parser.parseAttribute(res.emplace_back()); })
      || parser.parseRSquare()) {
    return failure();
  }
  args = parser.getBuilder().getArrayAttr(res);
  return success();
}
void printSBP(AsmPrinter& printer, ArrayAttr args) { printer << " " << args; }

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
