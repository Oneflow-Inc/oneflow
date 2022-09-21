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
#ifndef ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_ONEFLOW_LRJIT_REGISTRY_H_
#define ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_ONEFLOW_LRJIT_REGISTRY_H_

#include "oneflow/core/common/just.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/util.h"
#include "oneflow/ir/oneflow-extension/include/PyAst/Ast.h"

#include <cstdint>
#include <memory>
#include <unordered_map>
#include <utility>
#include <iostream>
#include <string>

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

  std::unordered_map<std::string, LRJITRegistry_Store_> functionId2engine_;
};

#endif  // ONEFLOW_IR_ONEFLOW_EXTENSION_INCLUDE_ONEFLOW_ONEFLOW_LRJIT_REGISTRY_H_
