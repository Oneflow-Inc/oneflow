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
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace test {

TEST(Maybe, JUST_MSG) {
  auto f = [](int x) -> Maybe<int> {
    if (x > 10) { return Error::ValueError("") << "input value " << x; }

    return 233;
  };

  auto g = [](int x) { return x * x - 5 * x + 3; };

  auto h = [&](int x) -> Maybe<int> {
    auto y = g(x);
    return JUST_MSG(f(y), "input value g(", x, ")");
  };

  auto data = std::mem_fn(&Maybe<int>::Data_YouAreNotAllowedToCallThisFuncOutsideThisFile);

  EXPECT_EQ(data(h(1)), 233);

  auto err = h(10).error();
  EXPECT_EQ(err->msg(), "input value 53");
  EXPECT_EQ(err->stack_frame(0).error_msg(), "(f(y)): input value g(10)");
}

}  // namespace test
}  // namespace oneflow
