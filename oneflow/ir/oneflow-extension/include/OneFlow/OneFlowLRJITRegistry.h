#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_AST_JIT_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_AST_JIT_H_

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <string>

#include "oneflow/core/common/just.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/util.h"
#include "oneflow/ir/oneflow-extension/include/PyAst/Ast.h"

namespace mlir {
class ExecutionEngine;
}

typedef std::pair<std::shared_ptr<mlir::ExecutionEngine>, std::function<double(double, double)>>
    LRJITRegistry_Store_;

class LRJITRegistry final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LRJITRegistry);
  ~LRJITRegistry() = default;

  void Register(const std::string& function_id, pyast::FunctionDef& ast, bool is_dump);
  std::function<double(double, double)> LookUp(const std::string& function_id);

 private:
  friend class oneflow::Singleton<LRJITRegistry>;
  LRJITRegistry() = default;

  std::unordered_map<std::string, LRJITRegistry_Store_> function_id2engine_;
};

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_AST_JIT_H_
