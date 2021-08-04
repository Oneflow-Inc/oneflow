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
#include "oneflow/core/common/cached_caller.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace test {

Maybe<int> Inc(int x) { return x + 1; }

Maybe<int> IncByConstRef(const int& x) { return x + 1; }

TEST(ThreadLocal, scalar) {
  int x = CHECK_JUST(ThreadLocal<Inc>(0));
  ASSERT_EQ(x, 1);
}

TEST(ThreadLocal, const_ref) {
  int x = CHECK_JUST(ThreadLocal<IncByConstRef>(0));
  ASSERT_EQ(x, 1);
}

namespace {

struct Foo {
  static Maybe<Foo> New(int x) { return std::shared_ptr<Foo>(new Foo{x}); }

  int x;
};

}  // namespace

TEST(ThreadLocal, _class) {
  const auto& foo = ThreadLocal<Foo::New>(10);
  const auto& bar = ThreadLocal<Foo::New>(10);
  ASSERT_EQ(foo->x, 10);
  ASSERT_TRUE(foo == bar);
}

}  // namespace test
}  // namespace oneflow
