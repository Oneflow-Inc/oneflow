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
#include "oneflow/core/ndarray/slice.h"
#include <gtest/gtest.h>

namespace oneflow {

namespace test {

TEST(Slice, size) {
  Slice slice({-2, 0, -1});
  slice.Bound(4);
  ASSERT_EQ(slice.Size(), 2);
}

TEST(Slice, contiguous) {
  Slice slice({0, -1, 1});
  slice.Bound(4);
  ASSERT_TRUE(slice.IsContiguous());
  ASSERT_FALSE(slice.IsCoveringAll());
}

TEST(Slice, is_covering_all) {
  Slice slice({});
  slice.Bound(4);
  ASSERT_TRUE(slice.IsCoveringAll());
  ASSERT_TRUE(slice.IsContiguous());
}

}  // namespace test

}  // namespace oneflow
