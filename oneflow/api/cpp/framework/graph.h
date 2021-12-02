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

#ifndef ONEFLOW_API_CPP_GRAPH_H_
#define ONEFLOW_API_CPP_GRAPH_H_

#include <memory>
#include <string>
#include <vector>
#include "oneflow/api/cpp/framework/device.h"
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/framework/tensor_tuple.h"
#include "oneflow/core/job/job.pb.h"
#include "oneflow/core/operator/op_conf.pb.h"

namespace oneflow {

class NNGraph;

}  // namespace oneflow

namespace oneflow_api {

class Graph final {
 public:
  explicit Graph(const std::string& model_path, const Device& device);
  explicit Graph(const std::string& model_path);
  std::vector<Tensor> Forward(const std::vector<Tensor>& inputs);
  void SetBatchSize(int batch_size);

  // not must, better if provided
  // void To(const Device& device);

 private:
  void Compile(const std::vector<Tensor>& inputs);
  std::vector<Tensor> Run(const std::vector<Tensor>& inputs);
  void AddOp(oneflow::OperatorConf op_conf);
  void BuildGraph(const std::vector<Tensor>& inputs);
  void LoadCheckpoint();
  void RegisterTensors();

  std::shared_ptr<oneflow::NNGraph> graph_ = nullptr;
  bool is_compiled_ = false;
  int batch_size_ = 0;
  Device device_;
  oneflow::Job job_;
  oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>> input_name_to_tensor_;
  oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>> output_name_to_tensor_;
  oneflow::HashMap<std::string, std::shared_ptr<oneflow::one::Tensor>> variable_op_name_to_tensor_;
};

Graph Load(const std::string& model_path, const Device& device);

Graph Load(const std::string& model_path);

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_GRAPH_H_
