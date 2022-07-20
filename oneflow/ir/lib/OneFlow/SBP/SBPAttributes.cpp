#include <iostream>
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
  if (succeeded(parser.parseOptionalRSquare())) {
    std::cout << "hello";
    args = parser.getBuilder().getArrayAttr({});
    return success();
  }
  llvm::SmallVector<Attribute> res;
  llvm::SmallVector<Attribute> nd_list;

  auto parserListElem = [&](llvm::SmallVector<Attribute>& list) {
    auto loc = parser.getCurrentLocation();
    if (failed(parser.parseAttribute(list.emplace_back()))) {
      parser.emitError(loc, "failed to parse an attribute here");
      return failure();
    }
    if (list.back().dyn_cast<sbp::SplitAttr>() || list.back().dyn_cast<sbp::BroadcastAttr>()
        || list.back().dyn_cast<sbp::PartialSumAttr>()) {
      return success();
    }
    parser.emitError(loc, "failed to parse a sbp attribute here");
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
        if (succeeded(parser.parseOptionalLSquare())) { return parserList(); }
        return parserListElem(res);
      })
      || parser.parseRSquare()) {
    return failure();
  }
  args = parser.getBuilder().getArrayAttr(res);
  return success();
}
void printSBP(AsmPrinter& printer, ArrayAttr args) { printer  << args; }

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
