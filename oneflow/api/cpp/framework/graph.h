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

#include "dtype.h"
#include "shape.h"
#include "device.h"
#include "ivalue.h"
#include "tensor.h"
#include <cstddef>
#include <string>
#include <functional>
#include <unordered_map>

namespace oneflow {

class NNGraph;

}  // namespace oneflow

namespace oneflow_api {

struct InputOutputAttribute {
  InputOutputAttribute(DType datatype, const Shape& input_output_shape, size_t input_output_index)
      : datatype_(datatype),
        input_output_shape_(input_output_shape),
        input_output_index_(input_output_index) {}
  InputOutputAttribute() : InputOutputAttribute(DType::kInvalidDataType, Shape(), 0) {}

  DType datatype_;
  Shape input_output_shape_;
  size_t input_output_index_;
};

using InputOutputInfos = std::unordered_map<std::string, InputOutputAttribute>;

class Graph {
 public:
  explicit Graph(const std::string& model_path, const Device& device = Device("cpu"));
  ~Graph();

  Graph(const Graph& graph) = delete;
  Graph(Graph&& graph) noexcept;

  Graph& operator=(const Graph& graph) = delete;
  Graph& operator=(Graph&& graph) noexcept;

  InputOutputInfos GetInputInfos();
  InputOutputInfos GetOutputInfos();
  IValue Forward(const IValue& inputs);
  void set_batch_size(int batch_size);

  void RegisterJobPass(const std::function<std::string(const std::string& job)>& pass_fn);

  static Graph Load(const std::string& model_path, const Device& device = Device("cpu"));

 private:
  class GraphImpl;
  std::unique_ptr<GraphImpl> graph_;
};

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_GRAPH_H_
