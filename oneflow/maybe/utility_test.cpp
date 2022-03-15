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

#include <gtest/gtest.h>
#include "oneflow/maybe/utility.h"

using namespace oneflow::maybe;

TEST(Utility, NullOpt) {
  NullOptType a, b(NullOpt), c(a);  // NOLINT

  a = NullOpt;

  a = b;

  ASSERT_EQ(a, NullOptType{});
  ASSERT_EQ(std::hash<NullOptType>()(a), std::hash<NullOptType>()(NullOpt));
  ASSERT_EQ(NullOpt, a);
  ASSERT_GE(NullOpt, a);
  ASSERT_LE(NullOpt, a);
  ASSERT_FALSE(NullOpt < a);
  ASSERT_FALSE(NullOpt > a);
}
