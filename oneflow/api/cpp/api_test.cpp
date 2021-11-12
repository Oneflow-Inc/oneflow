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
#include "oneflow/api/cpp/api.h"
#include "oneflow/api/cpp/device.h"

namespace oneflow_api {
namespace {

class EnvScope {  // NOLINT
 public:
  EnvScope() { initialize(); }
  ~EnvScope() { release(); }
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

}  // namespace oneflow_api
