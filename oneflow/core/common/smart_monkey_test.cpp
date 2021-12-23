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
#ifndef ENABLE_CHAOS
#define ENABLE_CHAOS
#endif  // ENABLE_CHAOS
#include "oneflow/core/common/smart_monkey.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace chaos {
namespace test {

bool ChaosBool() { return OF_CHAOS_BOOL_EXPR(true); }

TEST(chaos, simple) {
  MonkeyScope scope(std::make_unique<SmartMonkey>());
  ASSERT_FALSE(ChaosBool());
  ASSERT_TRUE(ChaosBool());
  ASSERT_TRUE(ChaosBool());
}

void RecusiveFailed(int n, const std::function<void()>& Check) {
  if (n <= 0) { return Check(); }
  OF_SMART_MONKEY_SOURCE_CODE_POS_SCOPE();
  RecusiveFailed(n - 1, Check);
}

TEST(chaos, recusive) {
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    for (int i = 0; i < 10; ++i) {
      RecusiveFailed(i, []() { ASSERT_FALSE(ChaosBool()); });
      RecusiveFailed(i, []() { ASSERT_TRUE(ChaosBool()); });
      RecusiveFailed(i, []() { ASSERT_TRUE(ChaosBool()); });
    }
  }
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    for (int i = 0; i < 10; ++i) {
      RecusiveFailed(i, []() { ASSERT_FALSE(ChaosBool()); });
      RecusiveFailed(i, []() { ASSERT_TRUE(ChaosBool()); });
      RecusiveFailed(i, []() { ASSERT_TRUE(ChaosBool()); });
    }
  }
}

TEST(chaos, OF_CHAOS_MODE_SCOPE) {
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    for (int i = 0; i < 10; ++i) {
      RecusiveFailed(i, []() {
        OF_CHAOS_MODE_SCOPE(false);
        ASSERT_TRUE(ChaosBool());
      });
      RecusiveFailed(i, []() { ASSERT_FALSE(ChaosBool()); });
      RecusiveFailed(i, []() { ASSERT_TRUE(ChaosBool()); });
      RecusiveFailed(i, []() { ASSERT_TRUE(ChaosBool()); });
    }
  }
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    for (int i = 0; i < 10; ++i) {
      RecusiveFailed(i, []() {
        OF_CHAOS_MODE_SCOPE(false);
        ASSERT_TRUE(ChaosBool());
      });
      RecusiveFailed(i, []() { ASSERT_FALSE(ChaosBool()); });
      RecusiveFailed(i, []() { ASSERT_TRUE(ChaosBool()); });
      RecusiveFailed(i, []() { ASSERT_TRUE(ChaosBool()); });
    }
  }
}

bool BinaryTreeFailed(int n) {
  if (n <= 0) { return ChaosBool(); }
  bool ret = true;
  {
    OF_SMART_MONKEY_SOURCE_CODE_POS_SCOPE();
    ret = BinaryTreeFailed(n - 1);
  }
  if (ret) {
    OF_SMART_MONKEY_SOURCE_CODE_POS_SCOPE();
    ret = BinaryTreeFailed(n - 1);
  }
  return ret;
}

TEST(chaos, binary_tree) {
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    ASSERT_FALSE(BinaryTreeFailed(0));
    ASSERT_TRUE(BinaryTreeFailed(0));
    ASSERT_TRUE(BinaryTreeFailed(0));
  }
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    ASSERT_FALSE(BinaryTreeFailed(1));
    ASSERT_FALSE(BinaryTreeFailed(1));
    ASSERT_TRUE(BinaryTreeFailed(1));
    ASSERT_TRUE(BinaryTreeFailed(1));
  }
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    ASSERT_FALSE(BinaryTreeFailed(2));
    ASSERT_FALSE(BinaryTreeFailed(2));
    ASSERT_FALSE(BinaryTreeFailed(2));
    ASSERT_FALSE(BinaryTreeFailed(2));
    ASSERT_TRUE(BinaryTreeFailed(2));
    ASSERT_TRUE(BinaryTreeFailed(2));
  }
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    ASSERT_FALSE(BinaryTreeFailed(3));
    ASSERT_FALSE(BinaryTreeFailed(3));
    ASSERT_FALSE(BinaryTreeFailed(3));
    ASSERT_FALSE(BinaryTreeFailed(3));
    ASSERT_FALSE(BinaryTreeFailed(3));
    ASSERT_FALSE(BinaryTreeFailed(3));
    ASSERT_FALSE(BinaryTreeFailed(3));
    ASSERT_FALSE(BinaryTreeFailed(3));
    ASSERT_TRUE(BinaryTreeFailed(3));
    ASSERT_TRUE(BinaryTreeFailed(3));
  }
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_FALSE(BinaryTreeFailed(4));
    ASSERT_TRUE(BinaryTreeFailed(4));
    ASSERT_TRUE(BinaryTreeFailed(4));
  }
  {
    MonkeyScope scope(std::make_unique<SmartMonkey>());
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_FALSE(BinaryTreeFailed(5));
    ASSERT_TRUE(BinaryTreeFailed(5));
    ASSERT_TRUE(BinaryTreeFailed(5));
  }
}

}  // namespace test
}  // namespace chaos
}  // namespace oneflow
