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

#include "oneflow/api/cpp/device.h"
#include "oneflow/api/cpp/graph.h"
#include "oneflow/api/cpp/tensor.h"
#include <cstdio>
#include <fstream>
#include <istream>
#include <memory>
#include <sstream>
#include <string>
#include <vector>
#include "oneflow/core/common/data_type.pb.h"
#include "oneflow/core/common/hash_container.h"
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/common/symbol.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/nn_graph.h"
#include "oneflow/core/framework/tensor.h"
#include "oneflow/core/graph/op_graph.h"
#include "oneflow/core/job/parallel_desc.h"
#include "oneflow/core/serving/saved_model.pb.h"

namespace oneflow_api {

Graph::Graph(const std::string& model_path, const Device& device) {
  std::string graph_name = "saved_model";
  graph_ = std::make_shared<oneflow::NNGraph>(graph_name);
  oneflow::Symbol<oneflow::Device> device_symbol =
      CHECK_JUST(oneflow::Device::New(device.type(), device.device_id()));
  CHECK_JUST(graph_->Load(model_path, device_symbol));
}

Graph::Graph(const std::string& model_path) : Graph(model_path, Device("cpu")) {}

void Graph::to(const Device& device) {}

std::vector<Tensor> Graph::forward(const std::vector<Tensor>& inputs) {
  return std::vector<Tensor>();
}

Graph load(const std::string& model_path, const Device& device) {
  Graph graph(model_path, device);
  return graph;
}

Graph load(const std::string& model_path) {
  Device device = Device("cpu");
  return load(model_path, device);
}

}  // namespace oneflow_api
