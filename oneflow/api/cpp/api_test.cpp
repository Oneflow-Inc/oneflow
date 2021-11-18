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

#include <mutex>
#include <gtest/gtest.h>
#include "oneflow/api/cpp/api.h"
#include "oneflow/api/cpp/device.h"
#include "oneflow/api/cpp/env.h"
#include "oneflow/api/cpp/tensor.h"

namespace oneflow_api {
namespace {
std::mutex g_mutex;

class EnvScope {  // NOLINT
 public:
  EnvScope() {
    initialize();
    g_mutex.lock();
  }
  ~EnvScope() {
    release();
    g_mutex.unlock();
  }
};

}  // namespace

TEST(Api, init_and_release) {
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
  Tensor tensor;

  Tensor tensor_with_device(device);

  Tensor tensor_with_shape_and_device(shape, device);

  ASSERT_EQ(tensor_with_shape_and_device.shape(), shape);
  ASSERT_EQ(tensor_with_shape_and_device.device(), device);

  tensor_with_shape_and_device.zeros_();
}

TEST(Api, tensor_from_blob) {
  EnvScope scope;

  double* data = new double[8];
  for (int i = 0; i < 8; ++i) { data[i] = i; }

  auto tensor = Tensor::from_blob(data, {2, 2, 2}, Device("cpu"));

  double* new_data = new double[8];
  Tensor::to_blob(tensor, new_data);
  for (int i = 0; i < 8; ++i) { ASSERT_EQ(new_data[i], data[i]); }

  delete[] data;
  delete[] new_data;
}

}  // namespace oneflow_api
