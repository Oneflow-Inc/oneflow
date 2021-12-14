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
#include <array>
#include <chrono>
#include <cstdint>
#include <functional>
#include <thread>
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

TEST(Api, graph_test) {
  EnvScope scope;

  const std::string file_name = __FILE__;
  const std::string directory = file_name.substr(0, file_name.rfind('/'));

  Graph graph = Load(directory + "/graph_test_model");
  std::vector<Tensor> inputs;
  inputs.emplace_back(Shape{2, 2});
  inputs[0].zeros_();

  Tensor output = graph.Forward(inputs).at(0);
  Shape shape = output.shape();
  ASSERT_EQ(shape.At(0), 2);
  ASSERT_EQ(shape.At(1), 2);
  std::array<float, 4> buf{};
  output.copy_to(buf.data());
  ASSERT_EQ(buf[0], 1);
  ASSERT_EQ(buf[1], 1);
  ASSERT_EQ(buf[2], 1);
  ASSERT_EQ(buf[3], 1);
}

TEST(Api, resnet_test) {
  EnvScope scope;

  Graph graph = Load("/home/zhouzekai/models/resnet50");
  std::vector<Tensor> inputs;
  inputs.emplace_back(Shape{1, 3, 224, 224});
  inputs[0].zeros_();

  Tensor output = graph.Forward(inputs).at(0);
  Shape shape = output.shape();
  ASSERT_EQ(shape.At(0), 1);
  ASSERT_EQ(shape.At(1), 1000);
  std::array<float, 1000> data{};
  output.copy_to(data.data());
  float expected_data[]{-1.07454,  -0.319766, -0.497719, -1.15014,  -0.677915,
                            -0.326854, -0.906118, 0.276201,  0.0704126, -0.519408};
  for (int i = 0; i < 10; i++) { ASSERT_NEAR(data[i], expected_data[i], 0.00001); }
}

TEST(Api, thread_test) {
  EnvScope scope;
  const Graph graphs[]{Load("/home/zhouzekai/models/resnet50"),
                       Load("/home/zhouzekai/models/resnet50"),
                       Load("/home/zhouzekai/models/resnet50")};
  auto graph_forward = [](Graph& graph) {
    std::vector<Tensor> inputs;
    inputs.emplace_back(Shape{1, 3, 224, 224});
    inputs[0].zeros_();
    for (int i = 0; i < 100; i++) {
      auto now = std::chrono::high_resolution_clock::now();
      Tensor output = graph.Forward(inputs).at(0);
      Shape shape = output.shape();
      ASSERT_EQ(shape.At(0), 1);
      ASSERT_EQ(shape.At(1), 1000);
      std::array<float, 1000> data{};
      output.copy_to(data.data());
      std::cout << std::this_thread::get_id() << " " << i << " "
                << std::chrono::duration_cast<std::chrono::milliseconds>(
                       std::chrono::high_resolution_clock::now() - now)
                       .count()
                << std::endl;
      float expected_data[]{-1.07454,  -0.319766, -0.497719, -1.15014,  -0.677915,
                            -0.326854, -0.906118, 0.276201,  0.0704126, -0.519408};
      for (int i = 0; i < 10; i++) { ASSERT_NEAR(data[i], expected_data[i], 0.00001); }
    }
  };

  std::thread threads[]{std::thread(std::bind(graph_forward, graphs[0])),
                        std::thread(std::bind(graph_forward, graphs[1])),
                        std::thread(std::bind(graph_forward, graphs[2]))};
  for (auto& thread : threads) { thread.join(); }
}

namespace {

inline Graph LoadGraph(const Device& device) {
  Graph graph = Load("/home/zhouzekai/models/large_linear", device);
  return graph;
}

inline void Forward(Graph& graph, const Device& device, int expected_batch_dim = 1) {
  std::vector<Tensor> inputs{Tensor(
      oneflow::one::functional::Rand(
          oneflow::Shape({expected_batch_dim, 5000}), oneflow::DType::Float(),
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
  graph.set_batch_size(10);
  Forward(graph, device, 10);
}

TEST(Api, graph_gpu_batching_test) {
  EnvScope scope;
  Device device("cuda", 0);
  Graph graph = LoadGraph(device);
  graph.set_batch_size(10);
  Forward(graph, device, 10);
}

TEST(Api, tensor_copy_test) {
  EnvScope scope;
  std::array<float, 4> data{};

  Tensor tensor(Shape{2, 2}, Device("cuda", 0));
  tensor.copy_to(data.data());

  Tensor tensor1(Shape{2, 2}, Device("cuda", 1));
  tensor.copy_to(data.data());

  Tensor tensor2(Shape{2, 2}, Device("cuda", 2));
  tensor.copy_to(data.data());

  Tensor tensor3(Shape{2, 2}, Device("cuda", 3));
  tensor.copy_to(data.data());
}

}  // namespace oneflow_api
