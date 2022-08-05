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

}  // namespace test
}  // namespace oneflow
