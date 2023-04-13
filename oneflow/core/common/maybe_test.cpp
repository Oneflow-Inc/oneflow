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
#include "oneflow/core/common/maybe.h"
#include "gtest/gtest.h"
#include <gtest/gtest-death-test.h>
#include <memory>
#include "oneflow/core/common/exception.h"
#include "oneflow/core/common/util.h"

namespace oneflow {
namespace test {

TEST(Maybe, JUST_MSG) {
  auto f = [](int x) -> Maybe<int> {
    if (x > 10) { return Error::InvalidValueError() << "input value " << x; }

    return 233;
  };

  auto g = [](int x) { return x * x - 5 * x + 3; };

  auto h = [&](int x) -> Maybe<int> {
    auto y = g(x);
    return JUST_MSG(f(y), "input value g(", x, ")");
  };

  auto i = [&](float x) -> Maybe<int> {
    int y = x;
    return JUST_MSG(h(y), std::stringstream() << "input value int(" << x << ")");
  };

  auto data = CHECK_JUST(i(1));
  ASSERT_EQ(data, 233);

  auto err = i(10.123).stacked_error();
  ASSERT_EQ(err->error_proto()->msg(), R"(input value 53)");
  ASSERT_GE(err->stack_frame().size(), 2);
  ASSERT_EQ(err->stack_frame().at(0)->code_text(), "f(y)");
  ASSERT_EQ(err->stack_frame().at(1)->code_text(), "h(y)");

  try {
    CHECK_JUST(i(10.234));
  } catch (const RuntimeException& e) {
    EXPECT_TRUE(std::string(e.what()).find(R"(input value 53)") != std::string::npos);
  }
}

TEST(Maybe, CHECK_OR_RETURN) {
  auto f = [](int x) -> Maybe<int> {
    CHECK_OR_RETURN(x > 10);
    return 233;
  };

  auto i = [&](float x) -> Maybe<int> { return JUST(f(x)); };

  auto data = CHECK_JUST(i(20));
  ASSERT_EQ(data, 233);

  auto err = i(1).stacked_error();
  ASSERT_GE(err->stack_frame().size(), 2);
  ASSERT_EQ(err->stack_frame().at(0)->code_text(), "CHECK_OR_RETURN(x > 10)");
  ASSERT_EQ(err->stack_frame().at(1)->code_text(), "f(x)");
}

TEST(Maybe, CHECK_OK) {
  auto f = [](int x) -> Maybe<int> {
    if (x > 10) { return Error::InvalidValueError() << "input value " << x; }

    return 233;
  };

  auto g = [&](int x) -> Maybe<int> {
    auto y = JUST(f(x));
    return f(y);
  };

  // NOLINTNEXTLINE(cppcoreguidelines-avoid-goto)
  ASSERT_EXIT(CHECK_OK(g(11)), testing::KilledBySignal(SIGABRT), R"(g\(11\) is not OK)");
}

TEST(Maybe, Noncopyable) { Maybe<std::unique_ptr<int>> a{std::make_unique<int>(1)}; }

}  // namespace test
}  // namespace oneflow
