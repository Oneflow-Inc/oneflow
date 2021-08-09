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
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace test {

TEST(Optional, copy_constructor) {
  Optional<int64_t> a(0);
  std::vector<Optional<int64_t>> vec;
  vec.push_back(a);
  ASSERT_TRUE(vec[0].has_value());
  int64_t val = CHECK_JUST(vec[0].value());
  ASSERT_EQ(val, 0);
}

TEST(Optional, move_constructor) {
  Optional<int64_t> a(0);
  std::map<int64_t, Optional<int64_t>> map;
  map.emplace(0, a);
  ASSERT_TRUE(map.at(0).has_value());
  int64_t val = CHECK_JUST(map.at(0).value());
  ASSERT_EQ(val, 0);
}

}  // namespace test
}  // namespace oneflow
