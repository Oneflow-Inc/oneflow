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

#include <gtest/gtest-death-test.h>
#include <gtest/gtest.h>
#include <memory>

#include "oneflow/maybe/maybe.h"
#include "oneflow/maybe/optional.h"

using namespace oneflow::maybe;

TEST(Just, MaybeBasic) {
  using Error = simple::StackedError<std::string>;
  using MaybeInt = Maybe<int, Error>;

  auto f = [](int x) -> MaybeInt {
    if (x > 10 || x < 0) { return Error{"not in range"}; }

    return x + 10;
  };

  auto g = [&f](int x) -> MaybeInt {
    if (x == 15) { return Error{"invalid value"}; }

    return JUST(f(x)) * 2;
  };

  auto h = [&g](int x) -> MaybeInt { return JUST(g(x)) + 2; };

  ASSERT_EQ(CHECK_JUST(h(0)), 22);

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(h(11)),
      R"(not in range.*(lambda|operator\(\)).*f\(x\).*(lambda|operator\(\)).*g\(x\).*TestBody.*h\(11\))");

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(h(15)), R"(invalid value.*(lambda|operator\(\)).*g\(x\).*TestBody.*h\(15\))");

  ASSERT_EQ(details::JustPrivateScope::StackedError(h(12)).StackSize(), 2);
  ASSERT_EQ(details::JustPrivateScope::StackedError(h(15)).StackSize(), 1);
}

TEST(Just, MaybeVoid) {
  using Error = simple::StackedError<std::string>;
  using MaybeVoid = Maybe<void, Error>;

  auto f = [](int& x) -> MaybeVoid {
    if (x > 10 || x < 0) { return Error{"not in range"}; }

    x = x + 5;
    return Ok;
  };

  auto g = [&f](int& x) -> MaybeVoid {
    if (x == 15) { return Error{"invalid value"}; }

    JUST(f(x));
    JUST(f(x));
    return Ok;
  };

  auto h = [&g](int& x) -> MaybeVoid {
    JUST(g(x));
    x = x + 2;
    return Ok;
  };

  int x = 0;
  CHECK_JUST(h(x));
  ASSERT_EQ(x, 12);

  x = 11;
  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(h(x)),
      R"(not in range.*(lambda|operator\(\)).*f\(x\).*(lambda|operator\(\)).*g\(x\).*TestBody.*h\(x\))");
  ASSERT_EQ(x, 11);

  x = 8;
  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(h(x)),
      R"(not in range.*(lambda|operator\(\)).*f\(x\).*(lambda|operator\(\)).*g\(x\).*TestBody.*h\(x\))");

  [[maybe_unused]] auto _ = h(x);  // NOLINT
  ASSERT_EQ(x, 13);
}

TEST(Just, MaybeRef) {
  using Error = simple::StackedError<std::string>;
  using MaybeRef = Maybe<const int&, Error>;

  int k = 100;

  auto f = [&k](const int& x) -> MaybeRef {
    if (x > 10 || x < 0) { return Error{"not in range"}; }

    if (x < 5) return x;
    return k;
  };

  auto g = [&f](const int& x) -> MaybeRef {
    if (x == 2) { return Error{"invalid value"}; }
    return JUST(f(x));
  };

  int x = 1;
  ASSERT_EQ(CHECK_JUST(g(x)), 1);

  const int& y = CHECK_JUST(g(5));
  ASSERT_EQ(y, 100);
  k = 200;
  ASSERT_EQ(y, 200);

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(g(11)), R"(not in range.*(lambda|operator\(\)).*f\(x\).*TestBody.*g\(11\))");

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(g(2)), R"(invalid value.*TestBody.*g\(2\))");
}

TEST(Just, MaybeErrorPtr) {
  using E = simple::StackedError<std::string>;
  using Error = std::unique_ptr<E>;
  using MaybeInt = Maybe<int, Error>;

  auto f = [](int x) -> MaybeInt {
    if (x > 10 || x < 0) { return std::make_unique<E>("not in range"); }

    return x + 10;
  };

  auto g = [&f](int x) -> MaybeInt {
    if (x == 15) { return std::make_unique<E>("invalid value"); }

    return JUST(f(x)) * 2;
  };

  auto h = [&g](int x) -> MaybeInt { return JUST(g(x)) + 2; };

  ASSERT_EQ(CHECK_JUST(h(0)), 22);

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(h(11)),
      R"(not in range.*(lambda|operator\(\)).*f\(x\).*(lambda|operator\(\)).*g\(x\).*TestBody.*h\(11\))");

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(h(15)), R"(invalid value.*(lambda|operator\(\)).*g\(x\).*TestBody.*h\(15\))");

  ASSERT_EQ(details::JustPrivateScope::StackedError(h(12))->StackSize(), 2);
  ASSERT_EQ(details::JustPrivateScope::StackedError(h(15))->StackSize(), 1);
}

namespace oneflow {
namespace maybe {

template<typename T>
struct JustTraits {
  template<typename U>
  static simple::StackedError<std::string> ValueNotFoundError(U&&) {
    return {"not found"};
  }

  template<typename U>
  static decltype(auto) Value(U&& v) {
    return *v;
  }
};

}  // namespace maybe
}  // namespace oneflow

TEST(Just, Optional) {
  using Error = simple::StackedError<std::string>;
  using MaybeInt = Maybe<int, Error>;

  Optional<int> a, b(1), c(2);

  auto f = [](const Optional<int>& x) -> MaybeInt {
    if (x == 1) return Error("hello");

    return JUST(x) + 1;
  };

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(f(a)), R"(not found.*(lambda|operator\(\)).*x.*TestBody.*f\(a\))");
  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(f(b)), R"(hello.*TestBody.*f\(b\))");

  ASSERT_EQ(CHECK_JUST(f(c)), 3);
}

TEST(Just, Ptr) {
  using Error = simple::StackedError<std::string>;
  using MaybeInt = Maybe<int, Error>;

  std::shared_ptr<int> a, b(std::make_shared<int>(1)), c(std::make_shared<int>(2));

  auto f = [](const std::shared_ptr<int>& x) -> MaybeInt {
    if (JUST(x) == 1) return Error("hello");

    return JUST(x) + 1;
  };

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(f(a)), R"(not found.*(lambda|operator\(\)).*x.*TestBody.*f\(a\))");
  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(f(b)), R"(hello.*TestBody.*f\(b\))");

  ASSERT_EQ(CHECK_JUST(f(c)), 3);
}

TEST(Just, WithMsg) {
  struct UniqueInt {
    int x;

    void drop() { x = -333; }

    explicit UniqueInt(int x) : x{x} {}
    UniqueInt(const UniqueInt& i) = delete;
    UniqueInt(UniqueInt&& i) noexcept : x{i.x} { i.drop(); }  // NOLINT
    UniqueInt& operator=(const UniqueInt& i) = delete;
    UniqueInt& operator=(UniqueInt&& i) noexcept {
      x = i.x;
      i.drop();
      return *this;
    }
    ~UniqueInt() { drop(); }
  };

  using Error = simple::StackedError<std::string>;
  using MaybeInt = Maybe<UniqueInt, Error>;

  auto f = [](UniqueInt x) -> MaybeInt {
    if (x.x > 10) { return Error{"input value " + std::to_string(x.x)}; }

    return UniqueInt{233};
  };

  auto g = [](UniqueInt x) {
    int y = x.x;
    return UniqueInt{y * y - 5 * y + 3};
  };

  auto h = [&](UniqueInt x) -> MaybeInt {
    int n = x.x;
    auto y = g(std::move(x));
    return JUST_MSG(f(std::move(y)), "input value g(", n, ")");
  };

  auto i = [&](float x) -> MaybeInt {
    UniqueInt y{int(x)};
    return JUST_MSG(h(std::move(y)), "input value int(", x, ")");
  };

  auto data = CHECK_JUST(i(1));
  ASSERT_EQ(data.x, 233);

  auto err = details::JustPrivateScope::StackedError(i(10.123));
  ASSERT_EQ(err.Error(), "input value 53");
  ASSERT_EQ(err.StackElem(0).message, "f(std::move(y)): input value g(10)");
  ASSERT_EQ(err.StackElem(1).message, "h(std::move(y)): input value int(10.123)");

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto)
  ASSERT_EXIT(CHECK_JUST(i(10.234)), testing::KilledBySignal(SIGABRT), R"(input value 53)");
}

TEST(Just, JustOpt) {
  auto f = [](int x) -> Optional<int> {
    if (x > 10) return NullOpt;

    return x + 1;
  };

  auto g = [&f](int x) -> Optional<int> { return OPT_JUST(f(x)) * 2; };

  ASSERT_EQ(CHECK_JUST(g(2)), 6);
  ASSERT_FALSE(g(11));

  auto h = [&](int x) -> Optional<int> {
    if (x == 10) return NullOpt;

    return OPT_JUST(g(x)) + OPT_JUST(f(x + 2));
  };

  ASSERT_FALSE(h(10));
  ASSERT_FALSE(h(9));
  ASSERT_EQ(h(8), 29);
}

TEST(Just, NoStack) {
  using Error = simple::NoStackError<std::string>;
  using MaybeInt = Maybe<int, Error>;

  auto f = [](int x) -> MaybeInt {
    if (x > 10 || x < 0) { return Error{"not in range"}; }

    return x + 10;
  };

  auto g = [&f](int x) -> MaybeInt {
    if (x == 15) { return Error{"invalid value"}; }

    return JUST(f(x)) * 2;
  };

  auto h = [&g](int x) -> MaybeInt { return JUST(g(x)) + 2; };

  ASSERT_EQ(CHECK_JUST(h(0)), 22);

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(h(11)), R"(not in range)");

  ASSERT_DEATH(  // NOLINT(cppcoreguidelines-avoid-goto)
      CHECK_JUST(h(15)), R"(invalid value)");

  ASSERT_EQ(details::JustPrivateScope::StackedError(h(12)).StackSize(), 0);
  ASSERT_EQ(details::JustPrivateScope::StackedError(h(15)).StackSize(), 0);
}
