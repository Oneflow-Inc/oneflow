#pragma once
#include <iostream>
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/Dialect.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Operation.h"
#include "mlir/IR/PatternMatch.h"
#include "mlir/InitAllTranslations.h"

#include "OneFlow/OneFlowDialect.h"
#include "OneFlow/OneFlowOps.h"
#include "mlir/Support/LLVM.h"

namespace mlir {
namespace oneflow {
class Generator {
 public:
  void run();
  explicit Generator(MLIRContext* context) : context(context), builder(context) {
    graph = ModuleOp::create(FileLineColLoc::get(builder.getStringAttr("file.mlir"), 0, 0));
    pdl = PDLPatternModule(
        ModuleOp::create(FileLineColLoc::get(builder.getStringAttr("pdl.mlir"), 0, 0)));
  }
  void dfs(int depth, SmallVector<Value*>& input);

 private:
  template<typename... Args>
  void dfs_broadcast_binary_ops(int depth, SmallVector<Value*>& input);
  template<typename T>
  void dfs_broadcast_binary_op(int depth, SmallVector<Value*>& input);

  const int max_depth = 3;
  MLIRContext* context;
  OpBuilder builder;
  ModuleOp graph;
  PDLPatternModule pdl;
  auto get_random_tensor();
};
}  // namespace oneflow
}  // namespace mlir
