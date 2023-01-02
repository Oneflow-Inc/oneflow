/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
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
        || list.back().dyn_cast<sbp::PartialSumAttr>() || list.back().dyn_cast<sbp::AnyAttr>()) {
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
void printSBP(AsmPrinter& printer, ArrayAttr args) { printer << args; }

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
