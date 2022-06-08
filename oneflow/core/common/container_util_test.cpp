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
#include "gtest/gtest.h"
#include "oneflow/core/common/container_util.h"

namespace oneflow {
namespace test {

TEST(VectorAt, write_int_vector) {
  std::vector<int> vec = {1, 2, 3, 4, 5};
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 1)), 2);
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 3)), 4);
  CHECK_JUST(VectorAt(vec, 1)) = 6;
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 1)), 6);
  CHECK_JUST(VectorAt(vec, 3)) = 8;
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 3)), 8);
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 0)), 1);
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 2)), 3);
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 4)), 5);
}

namespace {
class A {
 public:
  explicit A(int a) : a(a) {}
  int a;
};
}  // namespace

TEST(VectorAt, write_custom_class_vector) {
  std::vector<A> vec = {A(1), A(2)};
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 0)).a, 1);
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 1)).a, 2);
  CHECK_JUST(VectorAt(vec, 0)) = A(3);
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 0)).a, 3);
  CHECK_JUST(VectorAt(vec, 1)) = A(4);
  EXPECT_EQ(CHECK_JUST(VectorAt(vec, 1)).a, 4);
}

}  // namespace test
}  // namespace oneflow
