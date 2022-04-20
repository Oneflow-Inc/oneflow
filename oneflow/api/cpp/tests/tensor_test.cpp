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
#include "oneflow/api/cpp/tests/api_test.h"

namespace oneflow_api {

TEST(Api, device) {
  EnvScope scope;

  auto device = Device("cpu");
  ASSERT_EQ(device.type(), "cpu");

#ifdef WITH_CUDA
  device = Device("cuda:0");
  ASSERT_EQ(device.type(), "cuda");
  ASSERT_EQ(device.device_id(), 0);

  device = Device("cuda", 1);
  ASSERT_EQ(device.type(), "cuda");
  ASSERT_EQ(device.device_id(), 1);
#endif
}

TEST(Api, tensor) {
  EnvScope scope;

  const auto device = Device("cpu");
  const auto shape = RandomShape();
  const auto dtype = DType::kDouble;

  Tensor tensor;
  ASSERT_EQ(tensor.shape(), Shape());
  ASSERT_EQ(tensor.device(), Device("cpu"));
  ASSERT_EQ(tensor.dtype(), DType::kFloat);

  Tensor tensor_with_all(shape, device, dtype);

  ASSERT_EQ(tensor_with_all.shape(), shape);
  ASSERT_EQ(tensor_with_all.device(), device);
  ASSERT_EQ(tensor_with_all.dtype(), dtype);
}

TEST(Api, tensor_from_buffer_and_copy_to) {
  EnvScope scope;

  const auto shape = RandomShape();

#define TEST_TENSOR_FROM_AND_TO_BLOB(dtype, cpp_dtype)                                           \
  std::vector<cpp_dtype> data_##cpp_dtype(shape.Count(0)), new_data_##cpp_dtype(shape.Count(0)); \
  for (int i = 0; i < shape.Count(0); ++i) { data_##cpp_dtype[i] = i; }                          \
  auto tensor_##cpp_dtype =                                                                      \
      Tensor::from_buffer(data_##cpp_dtype.data(), shape, Device("cpu"), dtype);                 \
  tensor_##cpp_dtype.copy_to(new_data_##cpp_dtype.data());                                       \
  ASSERT_EQ(new_data_##cpp_dtype, data_##cpp_dtype);

  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kFloat, float)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kDouble, double)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt8, int8_t)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt32, int32_t)
  TEST_TENSOR_FROM_AND_TO_BLOB(DType::kInt64, int64_t)
}

TEST(Api, tensor_zeros) {
  EnvScope scope;

  const auto shape = RandomShape();

  std::vector<float> data(shape.Count(0)), target_data(shape.Count(0));

  Tensor tensor(shape, Device("cpu"), DType::kFloat);
  tensor.zeros_();
  tensor.copy_to(data.data());

  std::fill(target_data.begin(), target_data.end(), 0);

  ASSERT_EQ(data, target_data);
}

}  // namespace oneflow_api
