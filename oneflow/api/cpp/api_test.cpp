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
#include <algorithm>
#include <array>
#include <cstdint>
#include <random>
#include <thread>
#include <vector>
#include <gtest/gtest.h>
#include "oneflow/api/cpp/api.h"

namespace oneflow_api {
namespace {

class EnvScope {  // NOLINT
 public:
  EnvScope() { initialize(); }
  ~EnvScope() { release(); }
};

std::mt19937 rng(std::random_device{}());

Shape randomShape() {
  std::uniform_int_distribution<> dist_ndim(1, 4), dist_dims(16, 64);
  std::vector<std::int64_t> dims(dist_ndim(rng), 0);
  for (auto& x : dims) { x = dist_dims(rng); }
  return Shape(dims);
}

template<typename T>
std::vector<T> randomData(size_t size) {
  std::uniform_int_distribution<> dist(-100, 100);
  std::vector<T> data(size);
  for (auto& x : data) { x = static_cast<T>(dist(rng)); }
  return data;
}

template<typename T>
std::vector<T> relu(const std::vector<T>& data) {
  std::vector<T> result(data.begin(), data.end());
  T zero = static_cast<T>(0);
  for (auto& x : result) {
    if (x < zero) { x = zero; }
  }
  return result;
}

}  // namespace

TEST(Api, device) {
  EnvScope scope;

  auto device = Device("cpu");
  ASSERT_EQ(device.type(), "cpu");

#ifdef WITH_CUDA
  device = Device("cuda", 1);
  ASSERT_EQ(device.type(), "cuda");
  ASSERT_EQ(device.device_id(), 1);

  device = Device("cuda:2");
  ASSERT_EQ(device.type(), "cuda");
  ASSERT_EQ(device.device_id(), 2);
#endif
}

TEST(Api, tensor) {
  EnvScope scope;

  const auto device = Device("cpu");
  const auto shape = randomShape();
  const auto dtype = DType::kDouble;

  Tensor tensor;

  Tensor tensor_with_all(shape, device, dtype);

  ASSERT_EQ(tensor_with_all.shape(), shape);
  ASSERT_EQ(tensor_with_all.device(), device);
  ASSERT_EQ(tensor_with_all.dtype(), dtype);
}

TEST(Api, tensor_from_and_to_blob) {
  EnvScope scope;
#define TEST_TENSOR_FROM_AND_TO_BLOB(dtype, cpp_dtype)                             \
  std::array<cpp_dtype, 8> data_##cpp_dtype{}, new_data_##cpp_dtype{};             \
  for (int i = 0; i < 8; ++i) { data_##cpp_dtype[i] = i; }                         \
  auto tensor_##cpp_dtype =                                                        \
      Tensor::from_blob(data_##cpp_dtype.data(), {2, 2, 2}, Device("cpu"), dtype); \
  tensor_##cpp_dtype.copy_to(new_data_##cpp_dtype.data());                         \
  ASSERT_EQ(new_data_##cpp_dtype, data_##cpp_dtype);

  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kFloat, float)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kDouble, double)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt8, int8_t)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt32, int32_t)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt64, int64_t)
}

TEST(Api, tensor_zeros) {
  EnvScope scope;

  const auto shape = randomShape();

  std::vector<float> data(shape.Count(0)), target_data(shape.Count(0));

  Tensor tensor(shape, Device("cpu"), DType::kFloat);
  tensor.zeros_();
  tensor.copy_to(data.data());

  std::fill(target_data.begin(), target_data.end(), 0);

  ASSERT_EQ(data, target_data);
}

TEST(Api, nn_relu) {
  EnvScope scope;

  const auto testRelu = []() {
    const auto shape = randomShape();
    const auto data = randomData<float>(shape.Count(0));
    const auto target_data = relu(data);
    std::vector<float> result(shape.Count(0));

    auto tensor = Tensor::from_blob(data.data(), shape, Device("cpu"), DType::kFloat);
    auto result_tensor = nn::relu(tensor);

    result_tensor.copy_to(result.data());

    ASSERT_EQ(result, target_data);
  };

  testRelu();

  std::vector<std::thread> threads;
  std::uniform_int_distribution<> dist(8, 32);
  int n_threads = dist(rng);

  for (int i = 0; i < n_threads; ++i) { threads.emplace_back(std::thread(testRelu)); }

  for (auto& x : threads) { x.join(); }
}

}  // namespace oneflow_api
