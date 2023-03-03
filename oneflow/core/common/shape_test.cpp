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
#include "oneflow/core/common/shape.h"
#include "gtest/gtest.h"
#include <functional>
#include <algorithm>

namespace oneflow {

namespace test {

TEST(Shape, constructor_0) {
  Shape a;
  ASSERT_EQ(a.is_initialized(), false);
}

TEST(Shape, function_test_1) {
  Shape shape({4096, 16, 197, 197});
  ASSERT_EQ(shape.is_initialized(), true);
  ASSERT_EQ(shape.NumAxes(), 4);
  ASSERT_EQ(shape.elem_cnt(), 2543386624);
}

TEST(Shape, single_scalar_initializer_list) {
  Shape shape1({4});
  ASSERT_EQ(shape1.is_initialized(), true);
  ASSERT_EQ(shape1.NumAxes(), 1);
  ASSERT_EQ(shape1.elem_cnt(), 4);
  ASSERT_EQ(shape1[0], 4);

  Shape shape2{4};
  ASSERT_EQ(shape1, shape2);
  ASSERT_EQ(shape2.is_initialized(), true);
  ASSERT_EQ(shape2.NumAxes(), 1);
  ASSERT_EQ(shape2.elem_cnt(), 4);
  ASSERT_EQ(shape2[0], 4);
}

TEST(Dim, operators) {
  Dim dim(4);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim.is_known(), true);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim, 4);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim, Dim(4));
  std::cout << __LINE__ << std::endl;
  ASSERT_GT(dim, 3);
  std::cout << __LINE__ << std::endl;
  ASSERT_GT(dim, Dim(3));
  std::cout << __LINE__ << std::endl;
  ASSERT_LT(dim, 5);
  std::cout << __LINE__ << std::endl;
  ASSERT_LT(dim, Dim(5));
  std::cout << __LINE__ << std::endl;
  ASSERT_GE(dim, 4);
  std::cout << __LINE__ << std::endl;
  ASSERT_GE(dim, Dim(4));
  std::cout << __LINE__ << std::endl;
  ASSERT_LE(dim, 4);
  std::cout << __LINE__ << std::endl;
  ASSERT_LE(dim, Dim(4));
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim + 1, 5);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim + Dim(1), 5);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim - 1, 3);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim - Dim(1), 3);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim * 2, 8);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim * Dim(2), 8);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim / 2, 2);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim / Dim(2), 2);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim % 3, 1);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim % Dim(3), 1);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim / 3, 1);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim / Dim(3), 1);
  std::cout << __LINE__ << std::endl;

  dim = Dim::Unknown();
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim.is_known(), false);
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim, Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_NE(dim, 4);
  std::cout << __LINE__ << std::endl;
  ASSERT_NE(dim, Dim(4));
  std::cout << __LINE__ << std::endl;
  ASSERT_FALSE(dim > 3);
  std::cout << __LINE__ << std::endl;
  ASSERT_FALSE(dim > Dim(3));
  std::cout << __LINE__ << std::endl;
  std::cout << __LINE__ << std::endl;
  ASSERT_FALSE(dim < 3);
  std::cout << __LINE__ << std::endl;
  ASSERT_FALSE(dim < Dim(3));
  std::cout << __LINE__ << std::endl;
  ASSERT_FALSE(dim >= 3);
  std::cout << __LINE__ << std::endl;
  ASSERT_FALSE(dim >= Dim(3));
  std::cout << __LINE__ << std::endl;
  ASSERT_FALSE(dim <= 3);
  std::cout << __LINE__ << std::endl;
  ASSERT_FALSE(dim <= Dim(3));
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim + 1, Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim + Dim(1), Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim - 1, Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim - Dim(1), Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim * 2, Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim * Dim(2), Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim / 2, Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim / Dim(2), Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim % 3, Dim::Unknown());
  std::cout << __LINE__ << std::endl;
  ASSERT_EQ(dim % Dim(3), Dim::Unknown());
  std::cout << __LINE__ << std::endl;
}

}  // namespace test
}  // namespace oneflow
