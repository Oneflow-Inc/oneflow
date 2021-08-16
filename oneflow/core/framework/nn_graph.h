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
#ifndef ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_H_
#define ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_H_

#include "oneflow/core/framework/nn_graph_if.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/job/plan.pb.h"

namespace oneflow {

class Blob;
class Runtime;

class NNGraph final : public NNGraphIf {
 public:
  explicit NNGraph(const std::string& name) : name_(name), runtime_inited_(false) {}
  ~NNGraph();

  const std::string& job_name() const { return name_; }
  const std::vector<std::string>& inputs_op_names() const;
  const std::vector<std::string>& outputs_op_names() const;
  int64_t variable_op_size() const;

  Maybe<void> RegisterInputOpNames(const std::vector<std::string>& input_op_names);
  Maybe<void> RegisterOutputOpNames(const std::vector<std::string>& output_op_names);
  Maybe<void> RegisterVariableOpNamesAndTensors(
      const std::vector<std::string>& variable_op_names,
      const std::vector<std::shared_ptr<one::Tensor>>& variable_tensors);
  Maybe<void> CompileAndInitRuntime();

 private:
  Maybe<void> RegisterFreeEagerTensorsToVariableOpNames();
  void NewRuntimeBuffers();
  void CloseRuntimeBuffers();

  std::string name_;
  std::vector<std::string> input_op_names_;
  std::vector<std::string> output_op_names_;
  HashMap<std::string, Blob*> variable_op_name2eager_blob_;
  HashSet<std::string> variable_op_names_;
  Job job_;
  Plan plan_;
  // TODO(chengcheng): temp impl using runtime now, need reimplement for dynamic multi nn.Graph.
  std::unique_ptr<Runtime> runtime_;
  bool runtime_inited_;
};

Maybe<void> RunLazyNNGraph(const one::TensorTuple& inputs, const one::TensorTuple& outputs,
                           const one::TensorTuple& parameters,
                           const std::shared_ptr<NNGraph>& nn_graph);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_FRAMEWORK_NN_GRAPH_H_
