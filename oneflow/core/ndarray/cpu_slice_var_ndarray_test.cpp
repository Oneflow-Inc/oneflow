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

TEST(CpuSliceVarNdarray, one_elem_assign) {
  std::vector<int32_t> data({1});
  std::vector<int32_t> buffer({0});
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{1LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{1LL}, buffer.data());
  buffer_ndarray(0).CopyFrom(data_ndarray(0));
  ASSERT_EQ(data[0], buffer[0]);
}

TEST(CpuSliceVarNdarray, one_elem_assign_slice_on_slice) {
  std::vector<int32_t> data({1});
  std::vector<int32_t> buffer({0});
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{1LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{1LL}, buffer.data());
  buffer_ndarray(0)(0).CopyFrom(data_ndarray(0)(0));
  ASSERT_EQ(data[0], buffer[0]);
}

TEST(CpuSliceVarNdarray, 1d_assign) {
  std::vector<int32_t> data({0, 1, 2, 3, 4, 5, 6, 7, 8, 9});
  std::vector<int32_t> buffer(10, 0);
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{10LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{10LL}, buffer.data());
  buffer_ndarray({}).CopyFrom(data_ndarray({}));
  ASSERT_EQ(memcmp(data.data(), buffer.data(), sizeof(int32_t) * 10), 0);
}

TEST(CpuSliceVarNdarray, 1d_slice_assign) {
  std::vector<int32_t> data({1, 2, 3, 4, 5, 6, 7, 8});
  std::vector<int32_t> buffer(10, 100);
  std::vector<int32_t> expected({100, 1, 2, 3, 4, 5, 6, 7, 8, 100});
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{static_cast<int64_t>(data.size())}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{10LL}, buffer.data());
  ASSERT_EQ(buffer_ndarray({1, -1}).xpu_shape(), XpuShape(Shape({8})));
  buffer_ndarray({1, -1}).CopyFrom(data_ndarray({}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * 10), 0);
}

TEST(CpuSliceVarNdarray, 1d_slice) {
  std::vector<int32_t> data({100, 1, 2, 3, 4, 5, 6, 7, 8, 100});
  std::vector<int32_t> buffer(8, 100);
  std::vector<int32_t> expected({1, 2, 3, 4, 5, 6, 7, 8});
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{static_cast<int64_t>(data.size())}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{static_cast<int64_t>(buffer.size())}, buffer.data());
  buffer_ndarray({}).CopyFrom(data_ndarray({1, -1}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

TEST(CpuSliceVarNdarray, 2d_slice) {
  // clang-format off
  std::vector<int32_t> data({
      100, 100, 100, 100,
      100, 0,   1,   100,
      100, 2,   3,   100,
      100, 100, 100, 100,
  });
  // clang-format on
  std::vector<int32_t> buffer(4, 100);
  std::vector<int32_t> expected({0, 1, 2, 3});
  CpuNdarrayBuilder<int32_t, 2> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{4LL, 4LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{2LL, 2LL}, buffer.data());
  buffer_ndarray({}, {}).CopyFrom(data_ndarray({1, -1}, {1, -1}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

TEST(CpuSliceVarNdarray, 2d_slice_assign) {
  std::vector<int32_t> data({0, 1, 2, 3});
  std::vector<int32_t> buffer(16, 100);
  // clang-format off
  std::vector<int32_t> expected({
      100, 100, 100, 100,
      100, 0,   1,   100,
      100, 2,   3,   100,
      100, 100, 100, 100,
  });
  // clang-format on
  CpuNdarrayBuilder<int32_t, 2> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{2LL, 2LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{4LL, 4LL}, buffer.data());
  buffer_ndarray({1, -1}, {1, -1}).CopyFrom(data_ndarray({}, {}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

TEST(CpuSliceVarNdarray, 2d_slice_reverse) {
  // clang-format off
  std::vector<int32_t> data({
      100, 100, 100, 100,
      100, 0,   1,   100,
      100, 2,   3,   100,
      100, 100, 100, 100,
  });
  std::vector<int32_t> buffer(16, 100);
  std::vector<int32_t> expected({
      100, 100, 100, 100,
      100, 2,   3,   100,
      100, 0,   1,   100,
      100, 100, 100, 100,
  });
  // clang-format on
  CpuNdarrayBuilder<int32_t, 2> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{4LL, 4LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{4LL, 4LL}, buffer.data());
  buffer_ndarray({1, -1}, {1, -1}).CopyFrom(data_ndarray({-2, 0, -1}, {1, -1}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

TEST(CpuSliceVarNdarray, 3d_slice) {
  // clang-format off
  std::vector<int32_t> data({
      100, 100, 100, 100,
      100, 0,   1,   100,
      100, 2,   3,   100,
      100, 100, 100, 100,
	
      100, 100, 100, 100,
      100, 4,   5,   100,
      100, 6,   7,   100,
      100, 100, 100, 100,
  });
  std::vector<int32_t> buffer(8, -1);
  std::vector<int32_t> expected({
      0, 1,
      2, 3,

      4, 5,
      6, 7
  });
  // clang-format on
  CpuNdarrayBuilder<int32_t, 3> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{2LL, 4LL, 4LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{2LL, 2LL, 2LL}, buffer.data());
  buffer_ndarray.CopyFrom(data_ndarray({}, {1, -1}, {1, -1}));
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

TEST(CpuSliceVarNdarray, 3d_slice_assign) {
  // clang-format off
  std::vector<int32_t> data({
      0, 1,
      2, 3,

      4, 5,
      6, 7
  });
  std::vector<int32_t> buffer(32, 100);
  std::vector<int32_t> expected({
      100, 100, 100, 100,
      100, 0,   1,   100,
      100, 2,   3,   100,
      100, 100, 100, 100,
	
      100, 100, 100, 100,
      100, 4,   5,   100,
      100, 6,   7,   100,
      100, 100, 100, 100,
  });
  // clang-format on
  CpuNdarrayBuilder<int32_t, 3> ndarray;
  auto&& data_ndarray = ndarray.Var(Shape{2LL, 2LL, 2LL}, data.data());
  auto&& buffer_ndarray = ndarray.Var(Shape{2LL, 4LL, 4LL}, buffer.data());
  buffer_ndarray({}, {1, -1}, {1, -1}).CopyFrom(data_ndarray);
  ASSERT_EQ(memcmp(expected.data(), buffer.data(), sizeof(int32_t) * buffer.size()), 0);
}

}  // namespace test

}  // namespace oneflow
