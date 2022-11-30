#pragma once
#include <iostream>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllTranslations.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"

namespace mlir {
namespace oneflow {
class Generator {
 public:
  void run();
  explicit Generator(MLIRContext* context) : context(context), builder(context) {
    mop = ModuleOp::create(FileLineColLoc::get(builder.getStringAttr("file.mlir"), 0, 0));
    pdl = PDLPatternModule(mop);
  }

 private:
  MLIRContext* context;
  OpBuilder builder;
  ModuleOp mop;
  PDLPatternModule pdl;
  auto get_random_tensor();
};
}  // namespace oneflow
}  // namespace mlir
