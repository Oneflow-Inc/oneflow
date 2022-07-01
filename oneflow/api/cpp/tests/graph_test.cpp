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
#include <algorithm>
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <iostream>
#include <thread>
#include <vector>
#include "oneflow/api/cpp/framework.h"
#include "oneflow/api/cpp/framework/dtype.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/api/cpp/tests/api_test.h"

namespace oneflow_api {

namespace {

inline Graph LoadGraph(const Device& device) {
  Graph graph =
      Graph::Load("./oneflow/api/cpp/tests/graph_test_model/affine_with_parameter", device);
  return graph;
}

inline void Forward(Graph& graph, const Device& device, int expected_batch_dim = 1) {
  std::vector<float> data(expected_batch_dim * 3);
  std::fill(data.begin(), data.end(), 1);
  std::vector<Tensor> inputs;
  inputs.emplace_back(
      Tensor::from_buffer(data.data(), Shape({expected_batch_dim, 3}), device, DType::kFloat));
  const auto& value = graph.Forward(inputs);
  ASSERT_TRUE(value.IsTensor());
  Tensor output = value.ToTensor();
  Shape shape = output.shape();
  ASSERT_EQ(shape.At(0), expected_batch_dim);
  ASSERT_EQ(shape.At(1), 4);
  std::vector<float> buf(expected_batch_dim * 4);
  output.copy_to(buf.data());
  for (const float& element : buf) { ASSERT_EQ(element, 4); }
}

}  // namespace

TEST(Api, graph_cpu_test) {
  EnvScope scope;
  Device device("cpu");
  Graph graph = LoadGraph(device);
  Forward(graph, device, 1);
}

#ifdef WITH_CUDA
TEST(Api, graph_gpu_test) {
  EnvScope scope;
  Device device("cuda", 0);
  Graph graph = LoadGraph(device);
  Forward(graph, device);
}

TEST(Api, graph_multi_gpu_test) {
  EnvScope scope;
  Device device("cuda", 0);
  Graph graph = LoadGraph(device);
  Forward(graph, device);

  Device device1("cuda", 1);
  Graph graph1 = LoadGraph(device1);
  Forward(graph1, device1);
}
#endif

TEST(Api, graph_cpu_batching_test) {
  EnvScope scope;
  Device device("cpu");
  Graph graph = LoadGraph(device);
  graph.set_batch_size(10);
  Forward(graph, device, 10);
}

#ifdef WITH_CUDA
TEST(Api, graph_gpu_batching_test) {
  EnvScope scope;
  Device device("cuda", 0);
  Graph graph = LoadGraph(device);
  graph.set_batch_size(10);
  Forward(graph, device, 10);
}

TEST(Api, graph_multi_device_test) {
  EnvScope scope;
  Device device("cuda", 0);
  Graph graph = LoadGraph(device);
  Forward(graph, device, 1);

  Device device1("cuda", 1);
  Graph graph1 = LoadGraph(device1);
  Forward(graph1, device1, 1);

  Device device2("cpu");
  Graph graph2 = LoadGraph(device2);
  Forward(graph2, device2, 1);
}

TEST(Api, graph_unload_test) {
  {
    EnvScope scope;

    Device device("cuda", 0);
    Graph graph = LoadGraph(device);
    Forward(graph, device, 1);

    {
      Device device1("cuda", 1);
      Graph graph1 = LoadGraph(device1);
      Forward(graph1, device1, 1);
    }

    Device device2("cpu");
    Graph graph2 = LoadGraph(device2);
    Forward(graph2, device2, 1);
  }

  {
    EnvScope scope;

    Device device("cpu");
    Graph graph = LoadGraph(device);
    Forward(graph, device, 1);
  }
}
#endif

TEST(Api, graph_thread_test) {
  EnvScope scope;

  Device device("cpu");
  std::vector<Graph> graphs;
  for (int i = 0; i < 10; i++) { graphs.emplace_back(LoadGraph(device)); }

  std::vector<std::thread> threads;
  for (Graph& graph : graphs) {
    threads.emplace_back(std::thread(std::bind(Forward, std::move(graph), device, 1)));
  }
  for (auto& thread : threads) { thread.join(); }
}

TEST(Api, graph_input_order_test) {
  EnvScope scope;

  Device device("cpu");
  Graph graph = Graph::Load("./oneflow/api/cpp/tests/graph_test_model/affine_no_parameter", device);

  std::vector<Tensor> inputs;
  std::vector<float> x(3);
  std::fill(x.begin(), x.end(), 1);
  inputs.emplace_back(Tensor::from_buffer(x.data(), Shape({1, 3}), device, DType::kFloat));
  std::vector<float> a(3 * 2);
  std::fill(a.begin(), a.end(), 1);
  inputs.emplace_back(Tensor::from_buffer(a.data(), Shape({3, 2}), device, DType::kFloat));
  std::vector<float> b(2);
  std::fill(b.begin(), b.end(), 1);
  inputs.emplace_back(Tensor::from_buffer(b.data(), Shape({2}), device, DType::kFloat));

  const auto& value = graph.Forward(inputs);
  ASSERT_TRUE(value.IsTensor());
  Tensor output = value.ToTensor();
  Shape shape = output.shape();
  ASSERT_EQ(shape.At(0), 1);
  ASSERT_EQ(shape.At(1), 2);
  std::array<float, 2> buf{};
  output.copy_to(buf.data());
  ASSERT_EQ(buf[0], 4);
  ASSERT_EQ(buf[1], 4);
}

TEST(Api, graph_input_output_infos_test) {
  EnvScope scope;
  Device device("cpu");
  Graph graph = LoadGraph(device);

  auto input_infos = graph.GetInputInfos();
  auto output_infos = graph.GetOutputInfos();

  ASSERT_EQ(input_infos.size(), 1);
  ASSERT_EQ(output_infos.size(), 1);

  auto it = input_infos.begin();
  DType dtype = it->second.datatype_;
  Shape shape = it->second.input_output_shape_;
  size_t order = it->second.input_output_index_;
  ASSERT_EQ(dtype, DType::kFloat);
  ASSERT_EQ(shape.NumAxes(), 2);
  ASSERT_EQ(shape.At(0), 1);
  ASSERT_EQ(shape.At(1), 3);
  ASSERT_EQ(order, 0);

  it = output_infos.begin();
  dtype = it->second.datatype_;
  shape = it->second.input_output_shape_;
  order = it->second.input_output_index_;
  ASSERT_EQ(dtype, DType::kFloat);
  ASSERT_EQ(shape.NumAxes(), 2);
  ASSERT_EQ(shape.At(0), 1);
  ASSERT_EQ(shape.At(1), 4);
  ASSERT_EQ(order, 0);
}

}  // namespace oneflow_api
