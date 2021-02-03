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
#include <vector>

#include "oneflow/core/framework/op_expr.h"
#include "oneflow/core/job/scope.h"

namespace oneflow {
namespace one {

struct OpInterpreterContext {
  Scope* scope;
  bool is_mirrored_strategy_enabled;
};

class OpInterpreter {
 public:
  OpInterpreter() = default;
  virtual ~OpInterpreter() = default;

  virtual std::vector<TensorRef> Interpret(const OpExpr& op, const std::vector<TensorRef>& inputs,
                                           OpInterpreterContext* ctx) = 0;
};

class LazyOpInterpreter : public OpInterpreter {
 public:
  LazyOpInterpreter() : OpInterpreter() {}
  ~LazyOpInterpreter() = default;

  std::vector<TensorRef> Interpret(const OpExpr& op, const std::vector<TensorRef>& inputs,
                                   OpInterpreterContext* ctx) override;
};

class EagerOpInterpreter : public OpInterpreter {
 public:
  EagerOpInterpreter() : OpInterpreter() {}
  ~EagerOpInterpreter() = default;

  std::vector<TensorRef> Interpret(const OpExpr& op, const std::vector<TensorRef>& inputs,
                                   OpInterpreterContext* ctx) override;
};

}  // namespace one
}  // namespace oneflow
