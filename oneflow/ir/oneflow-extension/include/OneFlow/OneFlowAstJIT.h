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

class PyAst final {};

class JIT_Engine;

class LR_JIT final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(LR_JIT);
  ~LR_JIT() = default;
  void Register(const std::string& function_id, const PyAst& ast);
  std::shared_ptr<JIT_Engine> LookUp(const std::string& function_id);
  double Invoke(std::shared_ptr<JIT_Engine> engine, double base_lr, int64_t step);

 private:
  friend class oneflow::Singleton<LR_JIT>;
  LR_JIT() = default;
  std::unordered_map<std::string, std::shared_ptr<JIT_Engine>> function_id2engine_;
};

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_AST_JIT_H_
