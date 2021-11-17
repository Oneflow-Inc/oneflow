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

namespace oneflow {

class NNGraph;

}  // namespace oneflow

namespace oneflow_api {

class Device;
class Tensor;

class Graph {
 public:
  explicit Graph(const std::string& model_path, const Device& device);
  explicit Graph(const std::string& model_path);

  void to(const Device& device);
  std::vector<Tensor> forward(const std::vector<Tensor>& inputs);

 private:
  std::shared_ptr<oneflow::NNGraph> graph_ = nullptr;
};

// TODO(zzk0): model_path is a single file or a directory, it depends on how parameters are stored
Graph load(const std::string& model_path, const Device& device);

Graph load(const std::string& model_path);

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_GRAPH_H_
