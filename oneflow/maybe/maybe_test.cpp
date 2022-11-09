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
#include "oneflow/maybe/error.h"
#include "oneflow/maybe/maybe.h"

using namespace oneflow::maybe;

TEST(Maybe, Basic) {
  using Error = simple::StackedError<int>;
  Maybe<int, Error> a{1}, b{a}, c{Error(2)}, d{c};

  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_FALSE(c);
  ASSERT_FALSE(d);

  ASSERT_EQ(details::MaybePrivateScope::Value(a), 1);
  ASSERT_EQ(details::MaybePrivateScope::Value(b), 1);

  a = 2;
  ASSERT_EQ(details::MaybePrivateScope::Value(a), 2);
  ASSERT_EQ(details::MaybePrivateScope::Value(b), 1);

  ASSERT_EQ(details::MaybePrivateScope::StackedError(c).Error(), 2);
  ASSERT_EQ(details::MaybePrivateScope::StackedError(d).Error(), 2);

  a = c;
  ASSERT_EQ(details::MaybePrivateScope::StackedError(a).Error(), 2);
}

TEST(Maybe, NonPOD) {
  using Error = simple::StackedError<std::string>;
  Maybe<std::shared_ptr<int>, Error> a{Ok, new int{1}}, b{a}, c{Error("test")}, d{c};

  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_FALSE(c);
  ASSERT_FALSE(d);

  ASSERT_EQ(details::MaybePrivateScope::Value(a).use_count(), 2);

  {
    Maybe<std::shared_ptr<int>, Error> x(a);

    ASSERT_EQ(details::MaybePrivateScope::Value(x).use_count(), 3);

    x = c;
    ASSERT_FALSE(x);

    x = a;
    ASSERT_EQ(details::MaybePrivateScope::Value(x).use_count(), 3);
  }

  ASSERT_EQ(details::MaybePrivateScope::Value(a).use_count(), 2);

  ASSERT_EQ(*details::MaybePrivateScope::Value(a), 1);
  *details::MaybePrivateScope::Value(a) = 2;
  ASSERT_EQ(*details::MaybePrivateScope::Value(a), 2);

  ASSERT_EQ(details::MaybePrivateScope::StackedError(c).Error(), "test");
  ASSERT_EQ(details::MaybePrivateScope::StackedError(c).StackSize(), 0);
}

TEST(Maybe, Reference) {
  using Error = simple::StackedError<std::string>;

  const int& n = 1;
  Maybe<const int&, Error> a{n}, b{a}, c{Error("test")}, d{c};

  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_FALSE(c);
  ASSERT_FALSE(d);

  ASSERT_EQ(details::MaybePrivateScope::Value(a), 1);

  int k = 2;

  a = k;
  ASSERT_EQ(details::MaybePrivateScope::Value(a), 2);

  k = 3;
  ASSERT_EQ(details::MaybePrivateScope::Value(a), 3);

  int x = 1;
  Maybe<int&, Error> e{x}, f{e}, g{Error("test")}, h{g};

  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_FALSE(c);
  ASSERT_FALSE(d);

  ASSERT_EQ(details::MaybePrivateScope::Value(e), 1);

  e = k;
  ASSERT_EQ(details::MaybePrivateScope::Value(e), 3);

  details::MaybePrivateScope::Value(e) = 4;
  ASSERT_EQ(k, 4);
}

TEST(Maybe, Void) {
  using Error = simple::StackedError<std::string>;
  Maybe<void, Error> a{Ok}, b{a}, c{Error("test")}, d{c};

  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_FALSE(c);
  ASSERT_FALSE(d);

  ASSERT_EQ(details::MaybePrivateScope::StackedError(c).Error(), "test");

  c = Error("hello");
  ASSERT_EQ(details::MaybePrivateScope::StackedError(c).Error(), "hello");

  a = c;
  ASSERT_EQ(details::MaybePrivateScope::StackedError(a).Error(), "hello");
}

TEST(Maybe, PtrError) {
  using PointedError = simple::StackedError<std::string>;
  using Error = std::unique_ptr<PointedError>;
  Maybe<int, Error> a{1}, c{InPlaceError, new PointedError("test")};

  ASSERT_TRUE(a);
  ASSERT_FALSE(c);

  ASSERT_EQ(details::MaybePrivateScope::StackedError(c)->Error(), "test");
}

TEST(Maybe, NoStack) {
  using Error = simple::NoStackError<std::string>;
  Maybe<int, Error> a{1}, b{a}, c{InPlaceError, "hello"}, d{c};

  ASSERT_TRUE(a);
  ASSERT_TRUE(b);
  ASSERT_FALSE(c);
  ASSERT_FALSE(d);

  a = c;
  ASSERT_FALSE(a);
}

TEST(Maybe, Monadic) {
  using Error = simple::NoStackError<std::string>;
  Maybe<int, Error> a{1}, b{InPlaceError, "hello"};

  auto x2 = [](int x) { return x * 2; };

  auto x2e2 = [](int x) -> Maybe<int, Error> {
    if (x == 4) return Error("test");
    return x * 2;
  };

  ASSERT_EQ(CHECK_JUST(a.Map(x2).Map(x2)), 4);
  ASSERT_FALSE(b.Map(x2).Map(x2));

  a = 1;
  ASSERT_EQ(CHECK_JUST(a.Bind(x2e2).Bind(x2e2)), 4);

  a = 2;
  ASSERT_EQ(CHECK_JUST(a.Bind(x2e2)), 4);
  ASSERT_EQ(a.Bind(x2e2).Bind(x2e2).GetError(), "test");

  a = 4;
  ASSERT_EQ(a.Bind(x2e2).GetError(), "test");
  ASSERT_EQ(a.Bind(x2e2).GetError(), "test");
}
