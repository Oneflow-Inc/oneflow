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

#ifdef WITH_CUDA
TEST(Api, embedding_test) {
  EnvScope scope;
  Device device("cuda");
  Graph graph = Graph::Load("/path/to/embedding", device);
  int64_t batch_size = 10000;
  int64_t num_features = 39;

  std::vector<int64_t> data(batch_size * num_features);
  std::fill(data.begin(), data.end(), 1);
  std::vector<Tensor> inputs;
  inputs.emplace_back(
      Tensor::from_buffer(data.data(), Shape({batch_size, num_features}), device, DType::kInt64));

  const auto& value = graph.Forward(inputs);

  ASSERT_TRUE(value.IsTensor());
  Tensor output = value.ToTensor();
  Shape shape = output.shape();
  ASSERT_EQ(shape.At(0), batch_size);
  ASSERT_EQ(shape.At(1), 1);

  std::vector<float> buf(batch_size);
  output.copy_to(buf.data());
}
#endif

}  // namespace oneflow_api
