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
#include <gtest/gtest.h>
#include "oneflow/api/cpp/framework/dtype.h"
#include "oneflow/api/cpp/framework/ivalue.h"
#include "oneflow/api/cpp/tests/api_test.h"

namespace oneflow_api {

namespace {

std::mt19937 rng(std::random_device{}());

}

TEST(Api, ivalue) {
  std::uniform_real_distribution<> dist(-100, 100);
  std::uniform_int_distribution<> dist_bool(0, 1);

  const auto v_int = static_cast<int>(dist(rng));
  ASSERT_EQ(IValue(v_int).ToInt(), v_int);

  const auto v_int64 = static_cast<int64_t>(dist(rng));
  ASSERT_EQ(IValue(v_int64).ToInt(), v_int64);

  const auto v_float = static_cast<float>(dist(rng));
  ASSERT_EQ(IValue(v_float).ToDouble(), v_float);

  const auto v_double = static_cast<double>(dist(rng));
  ASSERT_EQ(IValue(v_double).ToDouble(), v_double);

  const auto v_bool = static_cast<bool>(dist_bool(rng));
  ASSERT_EQ(IValue(v_bool).ToBool(), v_bool);
}

TEST(Api, ivalue_tensor) {
  EnvScope scope;

  const auto device = Device("cpu");
  const auto shape = RandomShape();
  const auto dtype = DType::kDouble;

  const IValue i_tensor(Tensor(shape, device, dtype));
  const auto& tensor = i_tensor.ToTensor();

  ASSERT_EQ(tensor.shape(), shape);
  ASSERT_EQ(tensor.device(), device);
  ASSERT_EQ(tensor.dtype(), dtype);
}

TEST(Api, ivalue_tensor_vector) {
  EnvScope scope;

  const auto device = Device("cpu");

  const std::vector<Tensor> v_tensor_vector{Tensor(RandomShape(), device, DType::kDouble),
                                            Tensor(RandomShape(), device, DType::kFloat)};
  const auto i_tensor = IValue(v_tensor_vector);
  const auto& tensor_vector = i_tensor.ToTensorVector();

  ASSERT_EQ(v_tensor_vector.size(), tensor_vector.size());

  for (size_t i = 0; i < tensor_vector.size(); ++i) {
    ASSERT_EQ(v_tensor_vector[i].device(), tensor_vector[i].device());
    ASSERT_EQ(v_tensor_vector[i].shape(), tensor_vector[i].shape());
    ASSERT_EQ(v_tensor_vector[i].dtype(), tensor_vector[i].dtype());
  }
}

TEST(Api, ivalue_copy) {
  EnvScope scope;

  const auto device = Device("cpu");
  const auto shape = RandomShape();
  const auto dtype = DType::kDouble;

  const IValue i_tensor(Tensor(shape, device, dtype));
  const auto i_tensor_a = i_tensor;  // NOLINT

  ASSERT_EQ(i_tensor_a.ToTensor().shape(), shape);
  ASSERT_EQ(i_tensor_a.ToTensor().device(), device);
  ASSERT_EQ(i_tensor_a.ToTensor().dtype(), dtype);

  IValue i_tensor_b;
  i_tensor_b = i_tensor;

  ASSERT_EQ(i_tensor_b.ToTensor().shape(), shape);
  ASSERT_EQ(i_tensor_b.ToTensor().device(), device);
  ASSERT_EQ(i_tensor_b.ToTensor().dtype(), dtype);
}

TEST(Api, ivalue_move) {
  EnvScope scope;

  const auto device = Device("cpu");
  const auto shape = RandomShape();
  const auto dtype = DType::kDouble;

  IValue i_tensor_a = IValue(Tensor(shape, device, dtype));
  IValue i_tensor_b = IValue(Tensor(shape, device, dtype));

  IValue i_tensor_c = std::move(i_tensor_a);
  ASSERT_EQ(i_tensor_c.ToTensor().shape(), shape);
  ASSERT_EQ(i_tensor_c.ToTensor().device(), device);
  ASSERT_EQ(i_tensor_c.ToTensor().dtype(), dtype);

  IValue i_tensor_d;
  i_tensor_d = std::move(i_tensor_b);
  ASSERT_EQ(i_tensor_d.ToTensor().shape(), shape);
  ASSERT_EQ(i_tensor_d.ToTensor().device(), device);
  ASSERT_EQ(i_tensor_d.ToTensor().dtype(), dtype);

  ASSERT_EQ(i_tensor_a.IsNone(), true);
  ASSERT_EQ(i_tensor_b.IsNone(), true);
}

}  // namespace oneflow_api
