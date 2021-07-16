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
#ifndef ONEFLOW_XRT_TVM_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_TVM_GRAPH_COMPILER_H_

#include "oneflow/xrt/graph_compiler.h"
#include <tvm/runtime/module.h>
#include <tvm/relay/expr.h>
#include <tvm/relay/op.h>
#include <tvm/relay/function.h>

namespace oneflow {
namespace xrt {
namespace of_tvm {

class TVMGraphCompiler final : public GraphCompiler::Impl {
 public:
  explicit TVMGraphCompiler(const std::string& name);
  virtual ~TVMGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(const XrtGraph* graph,
                                      const std::vector<Parameter>& entry_params,
                                      const std::vector<Parameter>& return_params,
                                      const std::vector<InputOutputAlias>& aliases) override;

 private:
  tvm::runtime::Module builder_;
};

}  // namespace of_tvm
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_TVM_GRAPH_COMPILER_H_