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
#include "oneflow/core/common/util.h"
#include "oneflow/core/intrusive/flat_msg.h"

namespace oneflow {

namespace {

template<int field_counter, typename WalkCtxType, typename FieldType>
struct DumpFieldName {
  static void Call(WalkCtxType* ctx, FieldType* field, const char* field_name) {
    ctx->emplace_back(field_name);
  }
};

template<typename T>
std::vector<std::string> GetFieldNames(T* flat_msg) {
  std::vector<std::string> field_names;
  flat_msg->template __WalkVerboseField__<DumpFieldName>(&field_names);
  return field_names;
}

template<typename T>
void CheckSoleFieldName(T* flat_msg, const std::string& expected) {
  const auto& field_names = GetFieldNames(flat_msg);
  ASSERT_EQ(field_names.size(), 1);
  ASSERT_EQ(field_names.at(0), expected);
}
// clang-format off
FLAT_MSG_BEGIN(TestOptional)
  FLAT_MSG_DEFINE_OPTIONAL(int32_t, bar);
FLAT_MSG_END(TestOptional)
// clang-format on

TEST(FlatMsg, optional) {
  static_assert(std::is_trivial<TestOptional>::value, "TestOptional is not trivial");
  FlatMsg<TestOptional> foo_box;
  auto& foo = *foo_box.Mutable();
  ASSERT_TRUE(!foo.has_bar());
  ASSERT_EQ(foo.bar(), 0);
  ASSERT_TRUE(GetFieldNames(&foo).empty());
  *foo.mutable_bar() = 9527;
  ASSERT_TRUE(foo.has_bar());
  ASSERT_EQ(foo.bar(), 9527);
  auto field_names = GetFieldNames(&foo);
  ASSERT_EQ(field_names.size(), 1);
  ASSERT_EQ(field_names.at(0), "bar_");
}

// clang-format off
FLAT_MSG_BEGIN(FooOneof)
  FLAT_MSG_DEFINE_ONEOF(type,
      FLAT_MSG_ONEOF_FIELD(int32_t, case_0)
      FLAT_MSG_ONEOF_FIELD(int64_t, case_1)
      FLAT_MSG_ONEOF_FIELD(TestOptional, bar));
FLAT_MSG_END(FooOneof)
// clang-format on

TEST(FlatMsg, oneof) {
  FlatMsg<FooOneof> foo_box;
  auto& foo = *foo_box.Mutable();
  ASSERT_TRUE(GetFieldNames(&foo).empty());
  ASSERT_TRUE(!foo.has_bar());
  ASSERT_EQ(foo.bar().bar(), 0);
  foo.mutable_case_0();
  CheckSoleFieldName(&foo, "case_0_");
  ASSERT_TRUE(foo.has_case_0());
  FooOneof::FLAT_MSG_ONEOF_CASE(type) x = foo.type_case();
  ASSERT_TRUE(x == FooOneof::FLAT_MSG_ONEOF_CASE_VALUE(case_0));
  *foo.mutable_case_1() = 9527;
  CheckSoleFieldName(&foo, "case_1_");
  ASSERT_TRUE(foo.has_case_1());
  ASSERT_EQ(foo.case_1(), 9527);
}

// clang-format off
FLAT_MSG_BEGIN(FooRepeated)
  FLAT_MSG_DEFINE_REPEATED(char, char_field, 1);
  FLAT_MSG_DEFINE_REPEATED(TestOptional, bar, 10);
FLAT_MSG_END(FooRepeated)
// clang-format on

TEST(FlatMsg, repeated) {
  FlatMsg<FooRepeated> foo_box;
  auto& foo = *foo_box.Mutable();
  ASSERT_EQ(foo.bar_size(), 0);
  ASSERT_EQ(foo.bar().size(), 0);
  auto* bar = foo.mutable_bar()->Add();
  ASSERT_TRUE(!bar->has_bar());
  ASSERT_EQ(foo.bar_size(), 1);
  ASSERT_EQ(foo.bar().size(), 1);
  bar->set_bar(9527);
  ASSERT_TRUE(bar->has_bar());
  ASSERT_EQ(bar->bar(), 9527);
  bar = foo.mutable_bar()->Add();
  ASSERT_TRUE(!bar->has_bar());
  ASSERT_EQ(foo.bar_size(), 2);
  ASSERT_EQ(foo.bar().size(), 2);
  bar->set_bar(9528);
  for (const auto& x : foo.bar()) { ASSERT_TRUE(x.has_bar()); }
  foo.clear_bar();
  ASSERT_EQ(foo.bar_size(), 0);
}

// clang-format off
template<int N>
FLAT_MSG_BEGIN(TestTemplateFlatMsg);
  FLAT_MSG_DEFINE_REPEATED(char, char_field, N);
FLAT_MSG_END(TestTemplateFlatMsg);
// clang-format on

TEST(FlatMsg, flat_msg_template) {
  FlatMsg<TestTemplateFlatMsg<1024>> foo;
  ASSERT_TRUE(foo.Get().char_field().empty());
}

}  // namespace

}  // namespace oneflow
