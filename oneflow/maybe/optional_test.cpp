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
#include <memory>
#include "oneflow/maybe/optional.h"

using namespace oneflow::maybe;

using Private = details::OptionalPrivateScope;

TEST(Optional, Scalar) {
  Optional<int> a, b(1), c(a), d(b), e(NullOpt), bb(InPlace, 1);

  static_assert(std::is_same<decltype(Private::Value(a)), int&>::value, "");

  ASSERT_TRUE(!a.HasValue());
  ASSERT_TRUE(b.HasValue());
  ASSERT_EQ(b.ValueOr(0), 1);
  ASSERT_TRUE(!c.HasValue());
  ASSERT_EQ(c.ValueOr(233), 233);
  ASSERT_TRUE(d.HasValue());
  ASSERT_EQ(d.ValueOr(0), 1);
  ASSERT_TRUE(!e.HasValue());

  a = b;
  ASSERT_TRUE(a.HasValue());
  ASSERT_EQ(a.ValueOr(0), 1);

  a = NullOpt;
  ASSERT_TRUE(!a.HasValue());

  a = 222;
  ASSERT_TRUE(a.HasValue());
  ASSERT_TRUE(a);
  ASSERT_EQ(a.ValueOr(1), 222);

  Private::Value(a) = 2333;
  ASSERT_EQ(a.ValueOr(1), 2333);

  Optional<const int> f, g(1);
  ASSERT_TRUE(!f.HasValue());
  ASSERT_TRUE(g.HasValue());
  ASSERT_EQ(g.ValueOr(2), 1);

  static_assert(std::is_same<decltype(Private::Value(f)), const int&>::value, "");

  f = 1;
  ASSERT_TRUE(f.HasValue());
  ASSERT_EQ(f.ValueOr(2), 1);
  ASSERT_EQ(Private::Value(f), 1);

  int x = 2;
  ASSERT_EQ(f.ValueOr(x), 1);

  ASSERT_EQ(f.Emplace(2), 2);
  ASSERT_EQ(Private::Value(f), 2);

  f.Reset();
  ASSERT_TRUE(!f);
}

TEST(Optional, NonScalar) {
  auto x = std::make_shared<int>(233);
  ASSERT_EQ(x.use_count(), 1);

  Optional<std::shared_ptr<int>> a, b(x), aa(a), aaa(InPlace, std::make_shared<int>(244));
  ASSERT_EQ(x.use_count(), 2);
  ASSERT_EQ(*Private::Value(b), 233);
  static_assert(std::is_same<decltype(Private::Value(b)), std::shared_ptr<int>&>::value, "");

  ASSERT_TRUE(!a.HasValue());
  ASSERT_TRUE(!aa.HasValue());

  Optional<std::shared_ptr<int>> c(a), d(b);
  ASSERT_TRUE(!c.HasValue());

  ASSERT_EQ(x.use_count(), 3);
  ASSERT_EQ(b, d);

  a = x;
  ASSERT_EQ(x.use_count(), 4);

  a = NullOpt;
  ASSERT_EQ(x.use_count(), 3);

  a = b;
  ASSERT_EQ(x.use_count(), 4);
  ASSERT_EQ(a, b);

  {
    Optional<std::shared_ptr<int>> e(a);  // NOLINT
    ASSERT_EQ(x.use_count(), 5);

    Optional<std::shared_ptr<int>> f;
    f = e;
    ASSERT_EQ(x.use_count(), 6);
  }

  ASSERT_EQ(x.use_count(), 4);
  *Private::Value(a) = 234;
  ASSERT_EQ(*x, 234);

  Optional<std::shared_ptr<int>> g(std::move(a));
  ASSERT_EQ(x.use_count(), 4);

  {
    Optional<std::shared_ptr<int>> h;
    ASSERT_TRUE(!h.HasValue());

    h = std::move(b);
    ASSERT_EQ(x.use_count(), 4);
  }

  ASSERT_EQ(x.use_count(), 3);

  Optional<const std::shared_ptr<int>> i(x);
  ASSERT_EQ(x.use_count(), 4);
  static_assert(std::is_same<decltype(Private::Value(i)), const std::shared_ptr<int>&>::value, "");

  i = NullOpt;
  ASSERT_EQ(x.use_count(), 3);

  i.Emplace(x);
  ASSERT_EQ(x.use_count(), 4);

  i.Reset();
  ASSERT_EQ(x.use_count(), 3);

  i.Emplace(std::move(x));
  ASSERT_EQ(Private::Value(i).use_count(), 3);

  struct A {
    int id;
    std::string name;
  };

  Optional<A> a1, a2{InPlace, 233, "oneflow"};

  ASSERT_FALSE(a1);
  ASSERT_TRUE(a2);

  ASSERT_EQ(a1, NullOpt);
  ASSERT_EQ(Private::Value(a2).id, 233);
  ASSERT_EQ(Private::Value(a2).name, "oneflow");
}

TEST(Optional, Reference) {
  int x = 233;

  Optional<int&> a, b(x), c(a), d(b);

  ASSERT_TRUE(!a);
  ASSERT_TRUE(b);
  ASSERT_TRUE(!c);
  ASSERT_TRUE(d);

  ASSERT_EQ(Private::Value(b), 233);
  ASSERT_EQ(Private::Value(d), 233);

  static_assert(std::is_same<decltype(Private::Value(b)), int&>::value, "");

  a = x;
  ASSERT_TRUE(a);
  ASSERT_EQ(Private::Value(a), 233);

  a = NullOpt;
  ASSERT_TRUE(!a);

  a = b;
  ASSERT_TRUE(a);
  ASSERT_EQ(Private::Value(a), 233);

  Private::Value(a) = 234;
  ASSERT_EQ(x, 234);

  Optional<const int&> e, f(x), g(e), h(f);

  ASSERT_TRUE(!e);
  ASSERT_TRUE(f);
  ASSERT_TRUE(!g);
  ASSERT_TRUE(h);
  ASSERT_NE(NullOpt, h);

  ASSERT_EQ(Private::Value(f), 234);
  ASSERT_EQ(Private::Value(h), 234);

  static_assert(std::is_same<decltype(Private::Value(h)), const int&>::value, "");

  e = x;
  ASSERT_TRUE(e);
  ASSERT_EQ(e, x);
  ASSERT_EQ(e, 234);
  ASSERT_EQ(Private::Value(e), 234);

  e = NullOpt;
  ASSERT_TRUE(!e);
  ASSERT_EQ(e, NullOpt);
}

TEST(Optional, Hash) {
  Optional<int> a, b(123);

  ASSERT_EQ(std::hash<decltype(a)>()(a), NullOptHash);
  ASSERT_EQ(std::hash<decltype(a)>()(b), std::hash<int>()(123));

  auto si = std::make_shared<int>(123);
  Optional<std::shared_ptr<int>> c, d(si);

  ASSERT_EQ(std::hash<decltype(c)>()(c), NullOptHash);
  ASSERT_EQ(std::hash<decltype(c)>()(d), std::hash<decltype(si)>()(si));

  int x = 233;
  Optional<int&> e, f(x);

  ASSERT_EQ(std::hash<decltype(e)>()(e), NullOptHash);
  ASSERT_EQ(std::hash<decltype(e)>()(f), std::hash<int*>()(&x));

  Optional<const int&> g;
  ASSERT_EQ(std::hash<decltype(g)>()(g), NullOptHash);
}

TEST(Optional, Compare) {
  Optional<int> a, b, c(-1), d(0), e(1), f(1);

  ASSERT_EQ(a, b);
  ASSERT_EQ(e, f);
  ASSERT_NE(a, d);
  ASSERT_LT(b, c);
  ASSERT_LE(b, c);
  ASSERT_LE(c, c);
  ASSERT_LT(c, d);
  ASSERT_LT(d, e);
  ASSERT_GT(e, d);
  ASSERT_GT(d, c);
  ASSERT_GT(c, b);
  ASSERT_GE(c, b);
  ASSERT_GE(a, b);

  int x = 0, y = 1, z = -1;
  ASSERT_NE(a, x);
  ASSERT_EQ(d, x);
  ASSERT_NE(x, c);
  ASSERT_EQ(z, c);
  ASSERT_LT(a, x);
  ASSERT_LT(c, x);
  ASSERT_LT(d, y);
  ASSERT_LT(z, f);
  ASSERT_LE(a, x);
  ASSERT_LE(d, x);
  ASSERT_GT(x, a);
  ASSERT_GT(x, c);
  ASSERT_GT(y, d);
  ASSERT_GT(f, z);
  ASSERT_GE(x, a);
  ASSERT_GE(x, d);

  std::set<Optional<int>> s{2, NullOpt, -1, 3, NullOpt, 2};

  ASSERT_EQ(s.size(), 4);

  auto iter = s.begin();
  ASSERT_EQ(*(iter++), NullOpt);
  ASSERT_EQ(*(iter++), -1);
  ASSERT_EQ(*(iter++), 2);
  ASSERT_EQ(*(iter++), 3);
}

TEST(Optional, Monadic) {
  Optional<int> a(1), b, c(2);
  ASSERT_EQ(a.Map([](int x) { return x + 1; }), c);
  ASSERT_EQ(b.Map([](int x) { return x + 1; }), b);
  ASSERT_EQ(a.Map([](int x) { return std::string(x + 1, 'a'); }).Map([](const auto& x) {
    return (int)x.size();
  }),
            c);
  ASSERT_EQ(a.Bind([](int x) -> Optional<float> {
               if (x < 10) {
                 return x * 1.1;
               } else {
                 return NullOpt;
               }
             })
                .Map([](float x) { return x - 1; })
                .Map([](float x) { return std::abs(x - 0.1) < 0.001; }),
            Optional<bool>(true));

  int x = 0;
  [[maybe_unused]] auto _ = b.OrElse([&] { x++; }).OrElse([&] { x *= 2; });
  ASSERT_EQ(x, 2);
  ASSERT_EQ(b.OrElse([] { return Optional<int>(3); }).Map([](int x) { return x - 1; }), c);
}
