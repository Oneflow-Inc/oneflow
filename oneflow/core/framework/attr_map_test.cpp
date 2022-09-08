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
#include "oneflow/core/framework/attr_map.h"
#include "oneflow/core/framework/attr_value.h"
#include "oneflow/core/framework/mutable_attr_map.h"

namespace oneflow {
namespace test {

TEST(AttrMap, basic) {
  auto& mut_attr_map = THREAD_CACHED_MUTABLE_ATTR_MAP("zero", "one", "zeros", "ones");
  mut_attr_map.SetAllAttrs(static_cast<int32_t>(0), static_cast<int64_t>(1),
                           std::vector<int32_t>{0}, std::vector<int64_t>{1});
  AttrMap attr_map(mut_attr_map);
  {
    const auto& val = CHECK_JUST(attr_map.GetAttr<int32_t>("zero"));
    ASSERT_EQ(val, 0);
  }
  {
    const auto& val = CHECK_JUST(attr_map.GetAttr<int64_t>("one"));
    ASSERT_EQ(val, 1);
  }
  {
    const auto& val = CHECK_JUST(attr_map.GetAttr<std::vector<int32_t>>("zeros"));
    ASSERT_EQ(val.size(), 1);
  }
  {
    const auto& val = CHECK_JUST(attr_map.GetAttr<std::vector<int32_t>>("zeros"));
    ASSERT_EQ(val.at(0), 0);
  }
  {
    const auto& val = CHECK_JUST(attr_map.GetAttr<std::vector<int64_t>>("ones"));
    ASSERT_EQ(val.size(), 1);
  }
  {
    const auto& val = CHECK_JUST(attr_map.GetAttr<std::vector<int64_t>>("ones"));
    ASSERT_EQ(val.at(0), 1);
  }
}

TEST(AttrMap, hash_value) {
  HashMap<AttrMap, int32_t> attr_map2int_value;
  auto& mut_attr_map = THREAD_CACHED_MUTABLE_ATTR_MAP("zero", "one", "zeros", "ones");
  mut_attr_map.SetAllAttrs(static_cast<int32_t>(0), static_cast<int64_t>(1),
                           std::vector<int32_t>{0}, std::vector<int64_t>{1});
  ASSERT_EQ(AttrMap(mut_attr_map).hash_value(), AttrMap(mut_attr_map).hash_value());
  ASSERT_TRUE(AttrMap(mut_attr_map) == AttrMap(mut_attr_map));
}

TEST(AttrMap, hash_map) {
  HashMap<AttrMap, int32_t> attr_map2int_value;
  auto& mut_attr_map = THREAD_CACHED_MUTABLE_ATTR_MAP("zero", "one", "zeros", "ones");
  attr_map2int_value[AttrMap(mut_attr_map)] = 0;
  ASSERT_EQ(attr_map2int_value.at(AttrMap(mut_attr_map)), 0);
  mut_attr_map.SetAttr<0>(static_cast<int32_t>(0));
  attr_map2int_value[AttrMap(mut_attr_map)] = 1;
  ASSERT_EQ(attr_map2int_value.at(AttrMap(mut_attr_map)), 1);
  mut_attr_map.SetAttr<1>(static_cast<int64_t>(1));
  attr_map2int_value[AttrMap(mut_attr_map)] = 2;
  ASSERT_EQ(attr_map2int_value.at(AttrMap(mut_attr_map)), 2);
  mut_attr_map.SetAttr<2>(std::vector<int32_t>{0});
  attr_map2int_value[AttrMap(mut_attr_map)] = 3;
  ASSERT_EQ(attr_map2int_value.at(AttrMap(mut_attr_map)), 3);
  mut_attr_map.SetAttr<3>(std::vector<int64_t>{1});
  attr_map2int_value[AttrMap(mut_attr_map)] = 4;
  ASSERT_EQ(attr_map2int_value.at(AttrMap(mut_attr_map)), 4);
}

}  // namespace test
}  // namespace oneflow
