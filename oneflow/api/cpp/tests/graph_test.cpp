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

#include <gtest/gtest.h>
#include <cstdint>
#include "oneflow/api/cpp/framework/device.h"
#include "oneflow/api/cpp/framework/dtype.h"
#include "oneflow/api/cpp/framework/graph.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/api/cpp/nn/functional/activation.h"
#include "oneflow/api/cpp/tests/api_test.h"
#include "oneflow/core/common/shape.h"
#include "oneflow/core/framework/device.h"
#include "oneflow/core/framework/dtype.h"
#include "oneflow/core/functional/functional_api.yaml.h"

namespace oneflow_api {

namespace {

inline Graph LoadGraph(const Device& device) {
  Graph graph = Load("/home/zhouzekai/models/job_linear/saved_model.pb", device);
  return graph;
}

inline void Forward(Graph& graph, const Device& device, int expected_batch_dim = 1) {
  std::vector<Tensor> inputs{Tensor(
      oneflow::one::functional::Rand(
          oneflow::Shape({1, 5000}), oneflow::DType::Float(),
          oneflow::Device::New(device.type(), device.device_id()).GetOrThrow(), nullptr, false)
          .GetPtrOrThrow())};
  std::vector<Tensor> outputs = graph.Forward(inputs);
  Shape shape = outputs.at(0).shape();
  ASSERT_EQ(outputs.size(), 1);
  ASSERT_EQ(shape.At(0), expected_batch_dim);
  ASSERT_EQ(shape.At(1), 100000);
}

}  // namespace

TEST(Api, graph_cpu_test) {
  EnvScope scope;
  Device device("cpu");
  Graph graph = LoadGraph(device);
  Forward(graph, device);
}

TEST(Api, graph_gpu_test) {
  EnvScope scope;
  Device device("cuda", 0);
  Graph graph = LoadGraph(device);
  Forward(graph, device);
}

TEST(Api, graph_openvino_test) {
  EnvScope scope;
  Device device("cpu");
  Graph graph = LoadGraph(device);
  graph.enable_openvino();
  Forward(graph, device);
}

TEST(Api, graph_trt_test) {
  EnvScope scope;
  Device device("cuda:0");
  Graph graph = LoadGraph(device);
  graph.enable_tensorrt();
  Forward(graph, device);
}

TEST(Api, graph_cpu_batching_test) {
  EnvScope scope;
  Device device("cpu");
  Graph graph = LoadGraph(device);
  Forward(graph, device, 10);
}

TEST(Api, graph_gpu_batching_test) {
  EnvScope scope;
  Device device("cuda", 0);
  Graph graph = LoadGraph(device);
  Forward(graph, device, 10);
}

}  // namespace oneflow_api
