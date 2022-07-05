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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/thread_local_guard.h"

namespace oneflow {
namespace test {

template<typename T>
void AssertCurrentValue(const T& value) {
  ThreadLocalGuard<T> guard(value);
  ASSERT_TRUE(ThreadLocalGuard<T>::HasCurrentValue());
  ASSERT_EQ(ThreadLocalGuard<T>::CurrentValue(), value);
}

template<typename T>
void Assert(const T& value0, const T& value1) {
  ASSERT_FALSE(ThreadLocalGuard<T>::HasCurrentValue());
  {
    ThreadLocalGuard<T> guard(value0);
    ASSERT_TRUE(ThreadLocalGuard<T>::HasCurrentValue());
  }
  {
    ThreadLocalGuard<T> guard(value0);
    ASSERT_TRUE(ThreadLocalGuard<T>::HasCurrentValue());
    ASSERT_EQ(ThreadLocalGuard<T>::CurrentValue(), value0);
  }
  {
    ThreadLocalGuard<T> guard(value1);
    ASSERT_TRUE(ThreadLocalGuard<T>::HasCurrentValue());
    ASSERT_EQ(ThreadLocalGuard<T>::CurrentValue(), value1);
  }
  {
    ThreadLocalGuard<T> guard(value0);
    ASSERT_TRUE(ThreadLocalGuard<T>::HasCurrentValue());
    ASSERT_EQ(ThreadLocalGuard<T>::CurrentValue(), value0);
    {
      ThreadLocalGuard<T> nested_guard(value1);
      ASSERT_TRUE(ThreadLocalGuard<T>::HasCurrentValue());
      ASSERT_EQ(ThreadLocalGuard<T>::CurrentValue(), value1);
    }
    ASSERT_EQ(ThreadLocalGuard<T>::CurrentValue(), value0);
  }
}

TEST(ThreadLocalGuard, bool) { Assert<bool>(true, false); }

}  // namespace test
}  // namespace oneflow
