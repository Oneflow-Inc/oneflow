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
#include <system_error>
#include "oneflow/maybe/error.h"

using namespace oneflow::maybe;
using namespace oneflow::maybe::simple;
using namespace std::string_literals;

namespace oneflow {
namespace maybe {

// test if StackedErrorTraits can be applied to some simple types
template struct StackedErrorTraits<StackedError<std::error_code>>;
template struct StackedErrorTraits<NoStackError<std::error_code>>;

}  // namespace maybe
}  // namespace oneflow

TEST(StackedError, SimpleStackedError) {
  StackedError<std::error_code, std::string_view> a(std::make_error_code(std::errc::timed_out));

  ASSERT_EQ(a.Error(), std::errc::timed_out);
  ASSERT_EQ(a.StackSize(), 0);

  const auto& ec = a.Error();
  ASSERT_DEATH(a.Abort(),  // NOLINT(cppcoreguidelines-avoid-goto)
               ec.category().name() + ":"s + std::to_string(ec.value()));

  [&a] { a.PushStack(__FILE__, __LINE__, __PRETTY_FUNCTION__, "hello"); }();

  struct SomeType {
    explicit SomeType(decltype(a)& a) {
      a.PushStack(__FILE__, __LINE__, __PRETTY_FUNCTION__, "hi");
    }
  } x(a);

  ASSERT_EQ(a.StackSize(), 2);
  ASSERT_DEATH(a.Abort(),  // NOLINT(cppcoreguidelines-avoid-goto)
               "(lambda|operator\\(\\)).*hello.*\n.*SomeType::SomeType.*hi");

  ASSERT_EQ(a.StackElem(0).message, "hello");
  ASSERT_EQ(a.StackElem(1).message, "hi");
}

TEST(StackedError, SimpleNoStackError) {
  NoStackError<std::error_code> a(std::make_error_code(std::errc::address_in_use));

  ASSERT_EQ(a.Error(), std::errc::address_in_use);
  ASSERT_EQ(a.StackSize(), 0);

  const auto& ec = a.Error();
  ASSERT_DEATH(a.Abort(),  // NOLINT(cppcoreguidelines-avoid-goto)
               ec.category().name() + ":"s + std::to_string(ec.value()));

  a.PushStack(__FILE__, __LINE__, __PRETTY_FUNCTION__, "hello");
  ASSERT_EQ(a.StackSize(), 0);
}
