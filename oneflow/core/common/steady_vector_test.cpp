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
#include "oneflow/core/common/steady_vector.h"

namespace oneflow {
namespace test {

void TestSteadyVector(int granularity) {
  CHECK_GT(granularity, 0);
  SteadyVector<int> vec;
  ASSERT_EQ(vec.size(), 0);
  for (int i = 0; i < (1 << granularity); ++i) {
    vec.push_back(i);
    ASSERT_EQ(vec.at(i), i);
    ASSERT_EQ(vec.size(), i + 1);
  }
}

TEST(SteadyVector, simple) { TestSteadyVector(6); }

}  // namespace test
}  // namespace oneflow
