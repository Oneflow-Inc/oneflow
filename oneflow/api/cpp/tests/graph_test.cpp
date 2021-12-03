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
#include "oneflow/api/cpp/framework/graph.h"
#include "oneflow/api/cpp/framework/shape.h"
#include "oneflow/api/cpp/framework/tensor.h"
#include "oneflow/api/cpp/tests/api_test.h"

namespace oneflow_api {

TEST(Api, graph_test) {
  EnvScope scope;

  Graph graph = Load("used_name.pb");
  std::vector<Tensor> inputs;
  inputs.emplace_back(Tensor(Shape{1, 3, 224, 224}));

  std::vector<Tensor> outputs = graph.Forward(inputs);
  Shape shape = outputs.at(0).shape();
  ASSERT_EQ(shape.At(0), 1);
  ASSERT_EQ(shape.At(1), 1000);
}

}  // namespace oneflow_api
