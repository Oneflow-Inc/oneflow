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
#include <array>
#include <cstdint>
#include <gtest/gtest.h>
#include "oneflow/api/cpp/api.h"
#include "oneflow/api/cpp/dtype.h"

namespace oneflow_api {
namespace {

class EnvScope {  // NOLINT
 public:
  EnvScope() { initialize(); }
  ~EnvScope() { release(); }
};

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
  const auto shape = Shape({16, 8, 224, 224});
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

  std::array<float, 8> data{}, target_data{};
  target_data.fill(0);

  Tensor tensor({2, 2, 2}, Device("cpu"), DType::kFloat);
  tensor.zeros_();

  tensor.copy_to(data.data());

  ASSERT_EQ(data, target_data);
}

TEST(Api, nn) {
  EnvScope scope;

  std::array<float, 8> data{-3, -2, -1, 0, 1, 2, 3, 4};
  std::array<float, 8> target_data{0, 0, 0, 0, 1, 2, 3, 4};
  std::array<float, 8> result_data{};

  auto tensor = Tensor::from_blob(data.data(), {2, 2, 2}, Device("cpu"), DType::kFloat);
  auto result = nn::relu(tensor);

  result.copy_to(result_data.data());

  ASSERT_EQ(result_data, target_data);
}

}  // namespace oneflow_api
