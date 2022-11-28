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
// include sstream first to avoid some compiling error
// caused by the following trick
// reference: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65899
#include <sstream>
#include "gtest/gtest.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/intrusive/force_standard_layout.h"

namespace oneflow {

namespace intrusive {

namespace test {

constexpr const int unstandard_value = 999;
constexpr const int standard_value = 666;

struct Unstandard {
 public:
  explicit Unstandard(int* ptr) : x(unstandard_value), ptr_(ptr) {}
  ~Unstandard() { *ptr_ = unstandard_value; }

  Unstandard(const Unstandard&) = default;
  Unstandard(Unstandard&&) = default;
  Unstandard& operator=(const Unstandard&) = default;
  Unstandard& operator=(Unstandard&&) = default;

  int* ptr() const { return ptr_; }
  void set_ptr(int* val) { ptr_ = val; }

  int x;

 private:
  int* ptr_;
};

TEST(ForceStandardLayout, default_constructor) {
  int value = standard_value;
  ForceStandardLayout<Unstandard> sl(&value);
  ASSERT_EQ(sl.Get().x, unstandard_value);
  ASSERT_EQ(sl.Get().ptr(), &value);
}

TEST(ForceStandardLayout, copy_constructor) {
  int value = standard_value;
  const ForceStandardLayout<Unstandard> const_sl(&value);
  ForceStandardLayout<Unstandard> sl(const_sl);  // NOLINT
  ASSERT_EQ(sl.Get().x, unstandard_value);
  ASSERT_EQ(sl.Get().ptr(), &value);
}

TEST(ForceStandardLayout, move_constructor) {
  int value = standard_value;
  ForceStandardLayout<Unstandard> old_sl(&value);
  ForceStandardLayout<Unstandard> sl(std::move(old_sl));
  ASSERT_EQ(sl.Get().x, unstandard_value);
  ASSERT_EQ(sl.Get().ptr(), &value);
}

TEST(ForceStandardLayout, copy_assign) {
  int value = standard_value;
  const ForceStandardLayout<Unstandard> const_sl(&value);
  ForceStandardLayout<Unstandard> sl = const_sl;  // NOLINT
  ASSERT_EQ(sl.Get().x, unstandard_value);
  ASSERT_EQ(sl.Get().ptr(), &value);
}

TEST(ForceStandardLayout, move_assign) {
  int value = standard_value;
  ForceStandardLayout<Unstandard> sl = ForceStandardLayout<Unstandard>(&value);
  ASSERT_EQ(sl.Get().x, unstandard_value);
  ASSERT_EQ(sl.Get().ptr(), &value);
}

TEST(ForceStandardLayout, destructor) {
  int value = standard_value;
  { ForceStandardLayout<Unstandard> sl(&value); }
  ASSERT_EQ(value, unstandard_value);
}

}  // namespace test

}  // namespace intrusive

}  // namespace oneflow
