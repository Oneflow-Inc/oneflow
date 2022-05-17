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
#include "gtest/gtest.h"
#include "oneflow/core/intrusive/struct_traits.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

struct OneflowTestNamespaceFoo {
  OneflowTestNamespaceFoo() : x(0), bar(0), const_bar(0) {}

  int x;
  int bar;
  const int const_bar;
};

TEST(StructField, mutable_struct_mutable_field) {
  OneflowTestNamespaceFoo foo;
  auto* bar = &foo.bar;
  auto* struct_ptr = INTRUSIVE_FIELD(OneflowTestNamespaceFoo, bar)::StructPtr4FieldPtr(bar);
  auto* field_ptr = INTRUSIVE_FIELD(OneflowTestNamespaceFoo, bar)::FieldPtr4StructPtr(&foo);
  ASSERT_EQ(struct_ptr, &foo);
  ASSERT_EQ(field_ptr, bar);
}

TEST(StructField, mutable_struct_const_field) {
  OneflowTestNamespaceFoo foo;
  auto* bar = &foo.const_bar;
  auto* struct_ptr = INTRUSIVE_FIELD(OneflowTestNamespaceFoo, const_bar)::StructPtr4FieldPtr(bar);
  auto* field_ptr = INTRUSIVE_FIELD(OneflowTestNamespaceFoo, const_bar)::FieldPtr4StructPtr(&foo);
  ASSERT_EQ(struct_ptr, &foo);
  ASSERT_EQ(field_ptr, bar);
}

TEST(StructField, const_struct_mutable_field) {
  const OneflowTestNamespaceFoo foo;
  auto* bar = &foo.bar;
  auto* struct_ptr = INTRUSIVE_FIELD(OneflowTestNamespaceFoo, bar)::StructPtr4FieldPtr(bar);
  auto* field_ptr = INTRUSIVE_FIELD(OneflowTestNamespaceFoo, bar)::FieldPtr4StructPtr(&foo);
  ASSERT_EQ(struct_ptr, &foo);
  ASSERT_EQ(field_ptr, bar);
}

TEST(StructField, const_struct_const_field) {
  const OneflowTestNamespaceFoo foo;
  auto* bar = &foo.const_bar;
  auto* struct_ptr = INTRUSIVE_FIELD(OneflowTestNamespaceFoo, const_bar)::StructPtr4FieldPtr(bar);
  auto* field_ptr = INTRUSIVE_FIELD(OneflowTestNamespaceFoo, const_bar)::FieldPtr4StructPtr(&foo);
  ASSERT_EQ(struct_ptr, &foo);
  ASSERT_EQ(field_ptr, bar);
}

struct X {
  int a;
  int b;
};

struct Y {
  int c;
  X d;
};

TEST(StructField, compose) {
  using BFieldInY = intrusive::ComposeStructField<INTRUSIVE_FIELD(Y, d), INTRUSIVE_FIELD(X, b)>;
  Y y{};
  int* field_b = &y.d.b;
  ASSERT_EQ(BFieldInY::FieldPtr4StructPtr(&y), field_b);
  ASSERT_EQ(BFieldInY::StructPtr4FieldPtr(field_b), &y);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
