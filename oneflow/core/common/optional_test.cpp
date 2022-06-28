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
#include "oneflow/core/common/just.h"
#include "oneflow/core/common/optional.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/common/exception.h"

namespace oneflow {
namespace test {

TEST(Optional, copy_constructor) {
  Optional<int64_t> a(0);
  std::vector<Optional<int64_t>> vec;
  vec.emplace_back(a);
  ASSERT_TRUE(vec[0].has_value());
  int64_t val = CHECK_JUST(vec[0]);
  ASSERT_EQ(val, 0);
}

TEST(Optional, move_constructor) {
  Optional<int64_t> a(0);
  std::map<int64_t, Optional<int64_t>> map;
  map.emplace(0, a);
  ASSERT_TRUE(map.at(0).has_value());
  int64_t val = CHECK_JUST(map.at(0));
  ASSERT_EQ(val, 0);
}

TEST(Optional, JUST) {
  Optional<int> a(233), b;

  ASSERT_EQ(a.value_or(0), 233);
  ASSERT_EQ(b.value_or(1), 1);

  auto f = [](const Optional<int>& v) -> Maybe<int> { return JUST(v); };

  ASSERT_EQ(CHECK_JUST(f(a)), 233);
  ASSERT_EQ(f(b).error()->msg(), "");

  auto g = [](const Optional<int>& v) -> Optional<int> { return JUST_OPT(v); };

  ASSERT_EQ(CHECK_JUST(g(a)), 233);

  a = 234;
  ASSERT_EQ(CHECK_JUST(a), 234);

  b = a;
  ASSERT_EQ(CHECK_JUST(b), 234);

  b.reset();
  ASSERT_EQ(b.value_or(1), 1);

  Optional<const int> c(233);
  ASSERT_EQ(CHECK_JUST(c), 233);
}

TEST(Optional, reference) {
  int x = 1, z = 0;
  Optional<int&> a(x), b;

  x = 2;
  ASSERT_EQ(CHECK_JUST(a), 2);
  ASSERT_EQ(b.value_or(z), 0);

  CHECK_JUST(a) = 3;
  ASSERT_EQ(x, 3);

  Optional<const int&> c(x);
  ASSERT_EQ(CHECK_JUST(c), 3);
}

TEST(Optional, non_scalar) {
  Optional<std::vector<int>> a(InPlaceConstruct, 10), b;
  CHECK_JUST(a)->at(1) = 1;

  ASSERT_EQ(CHECK_JUST(a)->size(), 10);
  ASSERT_EQ(CHECK_JUST(a)->at(1), 1);

  auto x = std::make_shared<std::vector<int>>(1);
  ASSERT_EQ(b.value_or(x), x);

  ASSERT_EQ(b.value_or(std::vector<int>{1, 2, 3}), (std::vector<int>{1, 2, 3}));
  ASSERT_EQ(b.value_or(*x), *x);
  ASSERT_EQ(a.value_or(*x), *CHECK_JUST(a));

  ASSERT_EQ(Optional<std::vector<int>>().value_or(*x), *x);
  ASSERT_EQ(Optional<std::vector<int>>().value_or(std::vector<int>{1, 2, 3}),
            (std::vector<int>{1, 2, 3}));

  Optional<const std::vector<int>> c(std::vector<int>{1, 2, 3});

  ASSERT_EQ(CHECK_JUST(c)->at(1), 2);
}

TEST(Optional, optional_just_error_throw) {
  ASSERT_THROW(  // NOLINT(cppcoreguidelines-avoid-goto)
      {
        ([]() -> Maybe<int> {
          Optional<int> a;
          return JUST(a);
        })()
            .GetOrThrow();
      },
      Exception);
}

TEST(Optional, monadic_operations) {
  Optional<int> a(1), b, c(2);
  ASSERT_EQ(a.map([](int x) { return x + 1; }), c);
  ASSERT_EQ(b.map([](int x) { return x + 1; }), b);
  ASSERT_EQ(a.map([](int x) { return std::string(x + 1, 'a'); }).map([](const auto& x) {
    return (int)x->size();
  }),
            c);
  ASSERT_EQ(a.bind([](int x) -> Optional<float> {
               if (x < 10) {
                 return x * 1.1;
               } else {
                 return NullOpt;
               }
             })
                .map([](float x) { return x - 1; })
                .map([](float x) { return std::abs(x - 0.1) < 0.001; }),
            Optional<bool>(true));

  int x = 0;
  b.or_else([&] { x++; }).or_else([&] { x *= 2; });
  ASSERT_EQ(x, 2);
  ASSERT_EQ(b.or_else([] { return Optional<int>(3); }).map([](int x) { return x - 1; }), c);
}

}  // namespace test
}  // namespace oneflow
