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
#include <gtest/gtest.h>
#include "oneflow/api/cpp/api.h"

namespace oneflow_api {
namespace {

class EnvScope {  // NOLINT
 public:
  EnvScope() { initialize(); }
  ~EnvScope() { release(); }
};

}  // namespace

EnvScope scope;

TEST(Api, device) {
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
  const auto device = Device("cpu");
  const auto shape = Shape({16, 8, 224, 224});
  const auto dtype = DType::kDouble;

  Tensor tensor;

  Tensor tensor_with_all(shape, device, dtype);

  ASSERT_EQ(tensor_with_all.shape(), shape);
  ASSERT_EQ(tensor_with_all.device(), device);
  ASSERT_EQ(tensor_with_all.dtype(), dtype);

  tensor_with_all.zeros_();
}

TEST(Api, tensor_from_blob) {
  std::array<double, 8> data{}, new_data{};

  for (int i = 0; i < 8; ++i) { data[i] = i; }

  auto tensor = Tensor::from_blob(data.data(), {2, 2, 2}, Device("cpu"), DType::kDouble);
  Tensor::to_blob(tensor, new_data.data());

  ASSERT_EQ(new_data, data);
}

}  // namespace oneflow_api
