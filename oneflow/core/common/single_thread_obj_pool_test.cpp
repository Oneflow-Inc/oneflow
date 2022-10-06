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
#include "oneflow/core/common/single_thread_obj_pool.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace obj_pool {
namespace test {

TEST(SingleThreadObjPool, naive) {
  auto pool = std::make_shared<SingleThreadObjPool<int>>();
  auto* ptr = pool->make_shared().get();
  ASSERT_EQ(ptr, pool->make_shared().get());
}

struct Int {  // NOLINT
  Int() : x(0) {}
  explicit Int(int val) : x(val) {}
  ~Int() { x = 0; }
  int x;
};

TEST(SingleThreadObjPool, enable_reconstruct) {
  auto pool = std::make_shared<SingleThreadObjPool<Int, kEnableReconstruct>>();
  (void)pool->make_shared(333);
  ASSERT_EQ(0, pool->make_shared()->x);
}

TEST(SingleThreadObjPool, disable_reconstruct) {
  auto pool = std::make_shared<SingleThreadObjPool<Int, kDisableReconstruct>>();
  int value = pool->make_shared(333)->x;
  ASSERT_EQ(value, pool->make_shared()->x);
}

}  // namespace test
}  // namespace obj_pool
}  // namespace oneflow
