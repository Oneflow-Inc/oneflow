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
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

TEST(BalancedSplitter, split_20_to_6_part) {
  BalancedSplitter splitter(20, 6);
  ASSERT_TRUE(splitter.At(0) == Range(0, 4));
  ASSERT_TRUE(splitter.At(1) == Range(4, 8));
  ASSERT_TRUE(splitter.At(2) == Range(8, 11));
  ASSERT_TRUE(splitter.At(3) == Range(11, 14));
  ASSERT_TRUE(splitter.At(4) == Range(14, 17));
  ASSERT_TRUE(splitter.At(5) == Range(17, 20));
}

TEST(BalancedSplitter, split_2_to_3_part) {
  BalancedSplitter splitter(2, 3);
  ASSERT_TRUE(splitter.At(0) == Range(0, 1));
  ASSERT_TRUE(splitter.At(1) == Range(1, 2));
  ASSERT_TRUE(splitter.At(2) == Range(2, 2));
}

}  // namespace oneflow
