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

TEST(CpuConcatVarNdarray, two_elem_concat) {
  std::vector<int32_t> x0_data{0};
  std::vector<int32_t> x1_data{1};
  std::vector<int32_t> buffer{-1, -1};
  std::vector<int32_t> expected{0, 1};
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto x0 = ndarray.Var(Shape{1LL}, x0_data.data());
  auto x1 = ndarray.Var(Shape{1LL}, x1_data.data());
  ndarray.Var(Shape{2LL}, buffer.data()).CopyFrom(ndarray.Concatenate({x0, x1}));
  ASSERT_EQ(memcmp(buffer.data(), expected.data(), sizeof(int32_t) * 2), 0);
}

TEST(CpuConcatVarNdarray, two_elem_concat_assign) {
  std::vector<int32_t> x0_data{-1};
  std::vector<int32_t> x1_data{-1};
  std::vector<int32_t> buffer{0, 1};
  CpuNdarrayBuilder<int32_t, 1> ndarray;
  auto x0 = ndarray.Var(Shape{1LL}, x0_data.data());
  auto x1 = ndarray.Var(Shape{1LL}, x1_data.data());
  ndarray.Concatenate({x0, x1}).CopyFrom(ndarray.Var(Shape{2LL}, buffer.data()));
  ASSERT_EQ(x0_data[0], 0);
  ASSERT_EQ(x1_data[0], 1);
}

TEST(CpuConcatVarNdarray, 2d_concat) {
  // clang-format off
 std::vector<int32_t> x0_data{
   0, 1, 2,
   5, 6, 7,
 };
 std::vector<int32_t> x1_data{
            3, 4,
            8, 9,
 };
 std::vector<int32_t> expected{
   0, 1, 2, 3, 4,
   5, 6, 7, 8, 9,
 };
 std::vector<int32_t> buffer(10, -1);
  // clang-format on
  CpuNdarrayBuilder<int32_t, 2> ndarray;
  auto x0 = ndarray.Var(Shape{2LL, 3LL}, x0_data.data());
  auto x1 = ndarray.Var(Shape{2LL, 2LL}, x1_data.data());
  ndarray.Var(Shape{2LL, 5LL}, buffer.data()).CopyFrom(ndarray.template Concatenate<1>({x0, x1}));
  ASSERT_EQ(memcmp(buffer.data(), expected.data(), sizeof(int32_t) * 10), 0);
}

TEST(CpuConcatVarNdarray, 2d_concat_assign) {
  // clang-format off
 std::vector<int32_t> x_data{
   0, 1, 2, 3, 4,
   5, 6, 7, 8, 9,
 };
 std::vector<int32_t> y0_buffer(6, -1);
 std::vector<int32_t> y1_buffer(4, -1);
 std::vector<int32_t> y0_expected{
   0, 1, 2,
   5, 6, 7,
 };
 std::vector<int32_t> y1_expected{
            3, 4,
            8, 9,
 };
  // clang-format on
  CpuNdarrayBuilder<int32_t, 2> ndarray;
  auto x = ndarray.Var(Shape{2LL, 5LL}, x_data.data());
  auto y0 = ndarray.Var(Shape{2LL, 3LL}, y0_buffer.data());
  auto y1 = ndarray.Var(Shape{2LL, 2LL}, y1_buffer.data());
  ndarray.template Concatenate<1>({y0, y1}).CopyFrom(x);
  ASSERT_EQ(memcmp(y0_buffer.data(), y0_expected.data(), sizeof(int32_t) * 6), 0);
  ASSERT_EQ(memcmp(y1_buffer.data(), y1_expected.data(), sizeof(int32_t) * 4), 0);
}

TEST(CpuConcatVarNdarray, 3d_concat) {
  // clang-format off
 std::vector<int32_t> x0_data{
   0, 1, 2,
   5, 6, 7,

   10,11,12,
   15,16,17 
 };
 std::vector<int32_t> x1_data{
            3, 4,
            8, 9,
	      
            13,14,
            18,19,
 };
 std::vector<int32_t> expected{
   0, 1, 2, 3, 4,
   5, 6, 7, 8, 9,
     
   10,11,12,13,14,
   15,16,17,18,19,
 };
 std::vector<int32_t> buffer(20, -1);
  // clang-format on
  CpuNdarrayBuilder<int32_t, 3> ndarray;
  auto x0 = ndarray.Var(Shape{2LL, 2LL, 3LL}, x0_data.data());
  auto x1 = ndarray.Var(Shape{2LL, 2LL, 2LL}, x1_data.data());
  ndarray.Var(Shape{2LL, 2LL, 5LL}, buffer.data())
      .CopyFrom(ndarray.template Concatenate<2>({x0, x1}));
  ASSERT_EQ(memcmp(buffer.data(), expected.data(), sizeof(int32_t) * 20), 0);
}

TEST(CpuConcatVarNdarray, 3d_concat_assign) {
  // clang-format off
 std::vector<int32_t> x_data{
   0, 1, 2, 3, 4,
   5, 6, 7, 8, 9,
     
   10,11,12,13,14,
   15,16,17,18,19,
 };
 std::vector<int32_t> y0_expected{
   0, 1, 2,
   5, 6, 7,

   10,11,12,
   15,16,17 
 };
 std::vector<int32_t> y1_expected{
            3, 4,
            8, 9,
     
            13,14,
            18,19,
 };
 std::vector<int32_t> y0_buffer(2*2*3, -1);
 std::vector<int32_t> y1_buffer(2*2*2, -1);
  // clang-format on
  CpuNdarrayBuilder<int32_t, 3> ndarray;
  auto x = ndarray.Var(Shape{2LL, 2LL, 5LL}, x_data.data());
  auto y0 = ndarray.Var(Shape{2LL, 2LL, 3LL}, y0_buffer.data());
  auto y1 = ndarray.Var(Shape{2LL, 2LL, 2LL}, y1_buffer.data());
  ndarray.template Concatenate<2>({y0, y1}).CopyFrom(x);
  ASSERT_EQ(memcmp(y0_buffer.data(), y0_expected.data(), sizeof(int32_t) * y0_expected.size()), 0);
  ASSERT_EQ(memcmp(y1_buffer.data(), y1_expected.data(), sizeof(int32_t) * y1_expected.size()), 0);
}

}  // namespace test

}  // namespace oneflow
