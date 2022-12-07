#pragma once
#include <llvm/ADT/Hashing.h>
#include <iostream>
#include <unordered_map>
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
  void dfs(int depth, SmallVector<Value*>& inputs);

 private:
  template<typename... Args>
  void dfs_broadcast_binary_ops(int depth, SmallVector<Value*>& inputs);
  template<typename T>
  void dfs_broadcast_binary_op(int depth, SmallVector<Value*>& inputs);

  const int max_depth = 3;
  MLIRContext* context;
  OpBuilder builder;
  ModuleOp graph;
  PDLPatternModule pdl;
  std::unordered_map<llvm::hash_code, Operation*> D;
  auto get_random_tensor();
  ModuleOp build_pdl_from_oneflow_op(Operation* op);
  size_t fingerprint(Operation*);
  Operation* pdl_temp;
};
}  // namespace oneflow
}  // namespace mlir
