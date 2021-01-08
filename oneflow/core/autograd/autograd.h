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
#include "oneflow/core/framework/tensor.h"

namespace oneflow {

namespace one {

class FunctionNode {
 public:
  FunctionNode() = delete;
  virtual ~FunctionNode() = default;
  // Be used to release grad for non-leaf nodes
  virtual std::vector<std::shared_ptr<Tensor>> out_tensors() = 0;
  virtual std::shared_ptr<std::function<void()>> func() = 0;
};

class AutogradEngine {
 public:
  AutogradEngine() = default;
  virtual ~AutogradEngine() = default;

  std::shared_ptr<FunctionNode> AddBackwardFuncPtrIf(std::function<void()> fn) {
    auto ptr = AddBackwardFuncPtr(fn);
    CHECK_GT(ptr.use_count(), 1) << "The returned shared_ptr must belong to AutogradEngine";
    return ptr;
  }

  virtual void Execute(std::shared_ptr<FunctionNode> func_ptr, bool retain_graph) = 0;

 protected:
  virtual std::shared_ptr<FunctionNode> AddBackwardFuncPtr(std::function<void()>) = 0;
};

class StackAutogradEngine final : public AutogradEngine {
 public:
  void Execute(std::shared_ptr<FunctionNode> func_ptr, bool retain_graph) override;

 private:
  std::shared_ptr<FunctionNode> AddBackwardFuncPtr(std::function<void()>) override;
  std::vector<FunctionNode> func_list;  // TODO: use other container instead of vector
};

}  // namespace one

}  // namespace oneflow
