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
#include <sstream>
#include "gtest/gtest.h"
#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/intrusive/object_pool.h"

namespace oneflow {

namespace intrusive {

namespace test {

namespace {

class IntrusiveFoo final  // NOLINT
    : public intrusive::Base,
      public intrusive::EnableObjectPool<IntrusiveFoo, kThreadUnsafeAndDisableDestruct> {  // NOLINT
 public:
  IntrusiveFoo() = default;  // NOLINT

  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

 private:
  intrusive::Ref intrusive_ref_;
};

TEST(ObjectPool_kThreadUnsafeAndDisableDestruct, append_to_pool) {
  ObjectPool<IntrusiveFoo, kThreadUnsafeAndDisableDestruct> object_pool;
  IntrusiveFoo* ptr = nullptr;
  { ptr = object_pool.make_shared().get(); }
  ASSERT_EQ(ptr, object_pool.make_shared().get());
}

TEST(ObjectPool_kThreadUnsafeAndDisableDestruct, recycle) {
  ObjectPool<IntrusiveFoo, kThreadUnsafeAndDisableDestruct> object_pool;
  auto* ptr = object_pool.make_shared().get();
  ASSERT_EQ(ptr, object_pool.make_shared().get());
}

}  // namespace
}  // namespace test
}  // namespace intrusive
}  // namespace oneflow
