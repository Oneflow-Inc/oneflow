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
#ifndef ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_
#define ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_

#include "oneflow/xrt/graph_compiler.h"
#include "oneflow/xrt/xla/ops/op_context.h"
#include "oneflow/xrt/xla/xla_executable.h"
#include "tensorflow/compiler/xla/client/local_client.h"
#include "tensorflow/compiler/xla/client/xla_builder.h"
#include "tensorflow/compiler/xla/service/service.h"

namespace oneflow {
namespace xrt {
namespace mola {

class XlaGraphCompiler : public GraphCompiler::Impl {
 public:
  explicit XlaGraphCompiler(const std::string& name) : GraphCompiler::Impl(name) {
    builder_ = std::make_unique<xla::XlaBuilder>(name);
    CHECK(builder_) << "Creating xla builder failed.";
  }

  virtual ~XlaGraphCompiler() = default;

  std::shared_ptr<Executable> Compile(const XrtGraph* graph,
                                      const std::vector<Parameter>& entry_params,
                                      const std::vector<Parameter>& return_params,
                                      const std::vector<InputOutputAlias>& aliases) override;

 private:
  std::shared_ptr<Executable> BuildExecutable(const std::vector<xla::Shape>& xla_input_shapes,
                                              const xla::Shape& xla_output_shape,
                                              const xla::XlaComputation& computation);

  void SetupKernelContextParam(const XrtNode* node, XlaOpContext::Param* context_param);

  void BuildComputation(const XrtGraph* graph, const std::vector<Argument>& return_args,
                        xla::Shape* output_shape, xla::XlaComputation* computation);

  void BuildEntryParameters(const std::vector<Parameter>& entry_params,
                            std::vector<xla::Shape>* input_shapes);

  void SetOpMetadata(const std::string& op_type, const std::string& op_name);

  void ClearOpMetadata();

  Argument ArgFromParameter(const Parameter& param);

 private:
  bool use_meta_data_ = true;

  std::unique_ptr<xla::XlaBuilder> builder_;

  util::Map<Argument, XlaValue> operands_;
  util::Map<std::string /* argument name */, Argument> arguments_;
};

}  // namespace mola
}  // namespace xrt
}  // namespace oneflow

#endif  // ONEFLOW_XRT_XLA_XLA_GRAPH_COMPILER_H_
