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

#include <random>
#include <thread>
#include <gtest/gtest.h>
#include "oneflow/api/cpp/tests/api_test.h"

namespace oneflow_api {

namespace {

std::mt19937 rng(std::random_device{}());

template<typename T>
std::vector<T> Relu(const std::vector<T>& data) {
  std::vector<T> result(data.begin(), data.end());
  T zero = static_cast<T>(0);
  for (auto& x : result) {
    if (x < zero) { x = zero; }
  }
  return result;
}

}  // namespace

void TestRelu() {
  const auto shape = RandomShape();
  const auto data = RandomData<float>(shape.Count(0));
  const auto target_data = Relu(data);
  std::vector<float> result(shape.Count(0));

  auto tensor = Tensor::from_buffer(data.data(), shape, Device("cpu"), DType::kFloat);
  auto result_tensor = nn::relu(tensor);

  result_tensor.copy_to(result.data());

  ASSERT_EQ(result, target_data);
}

TEST(Api, nn_relu) {
  EnvScope scope;

  TestRelu();
}

TEST(Api, nn_relu_multithreading) {
  EnvScope scope;

  std::vector<std::thread> threads;
  std::uniform_int_distribution<> dist(8, 32);
  int n_threads = dist(rng);

  for (int i = 0; i < n_threads; ++i) { threads.emplace_back(std::thread(TestRelu)); }

  for (auto& x : threads) { x.join(); }
}

}  // namespace oneflow_api
