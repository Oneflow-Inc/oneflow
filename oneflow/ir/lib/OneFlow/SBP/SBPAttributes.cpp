#include "OneFlow/SBP/SBPDialect.h"
#include "OneFlow/SBP/SBPAttributes.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/ADT/TypeSwitch.h"
#include "mlir/IR/Attributes.h"
#include "mlir/IR/DialectImplementation.h"
#include "mlir/Support/LogicalResult.h"

using namespace mlir;

LogicalResult parseSBP(AsmParser& parser, FailureOr<ArrayAttr>& args) {
  if (failed(parser.parseLSquare())) { return failure(); }
  if (succeeded(parser.parseOptionalRSquare())) { return success(); }
  llvm::SmallVector<Attribute> res;
  llvm::SmallVector<Attribute> nd_list;

  auto parserListElem = [&](llvm::SmallVector<Attribute>& list) {
    if (failed(parser.parseAttribute(list.emplace_back()))) { return failure(); }
    if (list.back().dyn_cast<sbp::SplitAttr>() || list.back().dyn_cast<sbp::BroadcastAttr>()
        || list.back().dyn_cast<sbp::PartialSumAttr>()) {
      return success();
    }
    return failure();
  };

  auto parserList = [&]() {
    nd_list.clear();
    if (parser.parseCommaSeparatedList([&]() { return parserListElem(nd_list); })
        || parser.parseRSquare()) {
      return failure();
    }
    res.emplace_back(parser.getBuilder().getArrayAttr(nd_list));
    return success();
  };

  if (parser.parseCommaSeparatedList([&]() {
        if (succeeded(parser.parseLSquare())) {
          return parserList();
        }
        return parserListElem(res);
      })
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
