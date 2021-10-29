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
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/obj_pool.h"

namespace oneflow {
namespace obj_pool {
namespace test {

TEST(ObjectPool, thread_local_recycle) {
  using IntPtrPool = ObjectPool<int, 2>;
  int* ptr = IntPtrPool::GetOrNew();
  IntPtrPool::Put(ptr);
  ASSERT_EQ(IntPtrPool::GetOrNew(), ptr);
}

TEST(ObjectPool, static_global_recycle) {
  using IntPtrPool = ObjectPool<int, 1>;
  int* ptr = IntPtrPool::GetOrNew();
  IntPtrPool::Put(IntPtrPool::GetOrNew());
  IntPtrPool::Put(ptr);
  ASSERT_EQ(IntPtrPool::GetOrNew(), ptr);
  ASSERT_NE(IntPtrPool::GetOrNew(), ptr);
}

TEST(obj_pool_make_shared, naive) {
  int* ptr = obj_pool::make_shared<int>().get();
  ASSERT_EQ(obj_pool::make_shared<int>().get(), ptr);
}

TEST(obj_pool_make_shared, recyled_more) {
  int* raw_ptr0 = nullptr;
  int* raw_ptr1 = nullptr;
  {
    auto ptr0 = obj_pool::make_shared<int>();
    raw_ptr0 = ptr0.get();
    auto ptr1 = obj_pool::make_shared<int>();
    raw_ptr1 = ptr1.get();
  }
  {
    auto ptr0 = obj_pool::make_shared<int>();
    ASSERT_EQ(ptr0.get(), raw_ptr0);
    auto ptr1 = obj_pool::make_shared<int>();
    ASSERT_EQ(ptr1.get(), raw_ptr1);
  }
}

}  // namespace test
}  // namespace obj_pool
}  // namespace oneflow
