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
#include "tensor.h"
#include "iostream"

namespace oneflow {

class NNGraph;

}  // namespace oneflow

namespace oneflow_api {

class Graph {
 private:
  class GraphImpl;

 public:
  explicit Graph(const std::string& model_path, const Device& device);
  explicit Graph(const std::string& model_path);
  explicit Graph(const std::shared_ptr<GraphImpl>& graph);
  std::vector<Tensor> Forward(const std::vector<Tensor>& inputs);
  void set_batch_size(int batch_size);
  void enable_openvino();
  void enable_tensorrt();

 private:
  std::shared_ptr<GraphImpl> graph_;
};

Graph Load(const std::string& model_path, const Device& device);

Graph Load(const std::string& model_path);

// TODO(zzk0): only for debug, remove this
inline void PrintTensor(const Tensor& tensor) {
  std::cout << tensor.shape().elem_cnt() << " " << tensor.device().type() << " "
            << tensor.device().device_id() << " ";
  for (int i = 0; i < tensor.shape().NumAxes(); ++i) { std::cout << tensor.shape().At(i) << " "; }
  std::cout << std::endl;
  // float* data = new float[tensor.shape().elem_cnt() * 4];
  // tensor.copy_to(data);
  // for (int i = 0; i < tensor.shape().elem_cnt(); ++i) { std::cout << data[i] << " "; }
  // std::cout << std::endl;
  // delete[] data;
}

}  // namespace oneflow_api

#endif  // ONEFLOW_API_CPP_GRAPH_H_
