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
#include "oneflow/api/cpp/api.h"
#include "oneflow/api/cpp/env.h"

namespace mlir {
namespace oneflow {
class Generator {
 public:
  void run();
  void examples();
  explicit Generator(MLIRContext* context) : context(context), builder(context) {
    graph = ModuleOp::create(FileLineColLoc::get(builder.getStringAttr("file.mlir"), 0, 0));
    pdl = PDLPatternModule(
        ModuleOp::create(FileLineColLoc::get(builder.getStringAttr("pdl.mlir"), 0, 0)));
    ::oneflow_api::initialize();
  }
  ~Generator() { ::oneflow_api::release(); }
  void dfs(int depth, SmallVector<Value>& inputs);

 private:
  template<typename... Args>
  void dfs_broadcast_binary_ops(int depth, SmallVector<Value>& inputs);
  template<typename T>
  void dfs_broadcast_binary_op(int depth, SmallVector<Value>& inputs);
  template<typename... Args>
  void dfs_binary_ops(int depth, SmallVector<Value>& inputs);
  template<typename T>
  void dfs_binary_op(int depth, SmallVector<Value>& inputs);
  ModuleOp build_pdl_from_oneflow_op(Operation* op);
  size_t fingerprint(Operation*);
  // the graph rewrites can be infered
  bool can_be_infered_from_existing_rewrites(Operation* a, Operation* b) const;
  const int max_depth = 3;
  MLIRContext* context;
  OpBuilder builder;
  ModuleOp graph;
  PDLPatternModule pdl;
  auto get_random_tensor();
  std::unordered_map<size_t, Operation*> D{};
  std::vector<std::pair<Operation*, Operation*>> rewrites{};
  Operation* pdl_temp{};
  // TODO
  const std::string device_name = "@0:0";
  const std::string device_tag = "cpu";
  const int hierarchy = 1;
};
}  // namespace oneflow
}  // namespace mlir
