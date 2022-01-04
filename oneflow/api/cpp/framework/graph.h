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

#include "device.h"
#include "ivalue.h"
#include "tensor.h"

namespace oneflow {

class NNGraph;

}  // namespace oneflow

namespace oneflow_api {

class Graph {
 public:
  explicit Graph(const std::string& model_path, const Device& device = Device("cpu"));
  ~Graph();

  Graph(const Graph& graph) = delete;
  Graph(Graph&& graph) noexcept;

  Graph& operator=(const Graph& graph) = delete;
  Graph& operator=(Graph&& graph) noexcept;

  IValue Forward(const IValue& inputs);
  void set_batch_size(int batch_size);
  void enable_tensorrt();

  static Graph Load(const std::string& model_path, const Device& device = Device("cpu"));

 private:
  class GraphImpl;
  std::unique_ptr<GraphImpl> graph_;
};

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_GRAPH_H_
