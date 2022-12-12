#pragma once
#include <llvm/ADT/APSInt.h>
#include <llvm/ADT/Hashing.h>
#include <iostream>
#include <unordered_map>
#include "mlir/IR/Attributes.h"
#include "mlir/IR/BuiltinAttributes.h"
#include "mlir/IR/BuiltinTypes.h"
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
  explicit Generator(MLIRContext* context)
      : context(context),
        builder(context),
        device_name(StringAttr::get(context, "device_name"),
                    ArrayAttr::get(context, StringAttr::get(context, "@0:0"))),
        device_tag(StringAttr::get(context, "device_tag"), StringAttr::get(context, "cpu")),
        hierarchy(StringAttr::get(context, "hierarchy"),
                  ArrayAttr::get(context, IntegerAttr::get(IntegerType::get(context, 64), 1))),
        scope_symbol_id(StringAttr::get(context, "scope_symbol_id"),
                        IntegerAttr::get(IntegerType::get(context, 64), 114514)) {
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
  bool same_via_subst(Operation*, Operation*) const;
  const int max_depth = 3;
  MLIRContext* context;
  OpBuilder builder;
  ModuleOp graph;
  PDLPatternModule pdl;
  auto get_random_tensor();
  std::unordered_map<size_t, Operation*> D{};
  // pair of two PDL ModuleOp, without rewrite block
  std::vector<std::pair<Operation*, Operation*>> rewrites{};
  Operation* pdl_temp{};
  // TODO
  const NamedAttribute device_name;
  const NamedAttribute device_tag;
  const NamedAttribute hierarchy;
  const NamedAttribute scope_symbol_id;

  NamedAttribute op_name() {
    static size_t op_index{};
    return {StringAttr::get(context, "op_name"),
            StringAttr::get(context, std::to_string(op_index))};
  }
};
}  // namespace oneflow
}  // namespace mlir
