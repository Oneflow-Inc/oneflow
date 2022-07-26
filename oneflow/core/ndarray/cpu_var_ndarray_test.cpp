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
#include "oneflow/core/ndarray/cpu_ndarray_builder.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

TEST(CpuVarNdarray, one_elem_assign) {
  std::vector<int32_t> data({1});
  std::vector<int32_t> buffer({0});
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{1LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{1LL}, buffer.data());
  buffer_ndarray.CopyFrom(data_ndarray);
  ASSERT_EQ(data[0], buffer[0]);
}

TEST(CpuVarNdarray, 1d_assign) {
  std::vector<int32_t> data({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::vector<int32_t> buffer(10, 0);
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{10LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{10LL}, buffer.data());
  buffer_ndarray.CopyFrom(data_ndarray);
  ASSERT_EQ(memcmp(data.data(), buffer.data(), sizeof(int32_t) * 10), 0);
}

}  // namespace test

}  // namespace oneflow
