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
#include "oneflow/core/intrusive/flat_msg_view.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

// clang-format off
FLAT_MSG_BEGIN(VariantFoo);
  FLAT_MSG_DEFINE_STRICT_ONEOF(_,
    FLAT_MSG_ONEOF_FIELD(int8_t, int8_value)
    FLAT_MSG_ONEOF_FIELD(int16_t, int16_value)
    FLAT_MSG_ONEOF_FIELD(int32_t, int32_value)
    FLAT_MSG_ONEOF_FIELD(float, float_value));
FLAT_MSG_END(VariantFoo);
// clang-format on

// clang-format off
FLAT_MSG_BEGIN(VariantList);
  FLAT_MSG_DEFINE_REPEATED(VariantFoo, foo, 16);
FLAT_MSG_END(VariantList);
// clang-format on

// clang-format off
FLAT_MSG_VIEW_BEGIN(ViewFoo);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int32_t, int32_value);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int16_t, int16_value);
  FLAT_MSG_VIEW_DEFINE_PATTERN(float, float_value);
FLAT_MSG_VIEW_END(ViewFoo);
// clang-format on

TEST(FlatMsgView, match_success) {
  FlatMsg<VariantList> variant_list;
  variant_list.Mutable()->mutable_foo()->Add()->set_int32_value(30);
  variant_list.Mutable()->mutable_foo()->Add()->set_int16_value(40);
  variant_list.Mutable()->mutable_foo()->Add()->set_float_value(50.0);
  FlatMsgView<ViewFoo> view;
  ASSERT_TRUE(view.template Match(variant_list.Get().foo()));
  ASSERT_EQ(view->int32_value(), 30);
  ASSERT_EQ(view->int16_value(), 40);
  ASSERT_EQ(view->float_value(), 50.0);
}

TEST(FlatMsgView, match_failed) {
  FlatMsg<VariantList> variant_list;
  variant_list.Mutable()->mutable_foo()->Add()->set_int16_value(40);
  variant_list.Mutable()->mutable_foo()->Add()->set_int32_value(30);
  variant_list.Mutable()->mutable_foo()->Add()->set_float_value(50.0);
  FlatMsgView<ViewFoo> view;
  ASSERT_TRUE(!view.template Match(variant_list.Get().foo()));
}

TEST(FlatMsgView, match_success_vector) {
  std::vector<FlatMsg<VariantFoo>> variant_list(3);
  variant_list.at(0)->set_int32_value(30);
  variant_list.at(1)->set_int16_value(40);
  variant_list.at(2)->set_float_value(50.0);
  FlatMsgView<ViewFoo> view;
  ASSERT_TRUE(view.template Match(variant_list));
  ASSERT_EQ(view->int32_value(), 30);
  ASSERT_EQ(view->int16_value(), 40);
  ASSERT_EQ(view->float_value(), 50.0);
}

TEST(FlatMsgView, match_failed_vector) {
  std::vector<FlatMsg<VariantFoo>> variant_list(3);
  variant_list.at(0)->set_int16_value(40);
  variant_list.at(1)->set_int32_value(30);
  variant_list.at(2)->set_float_value(50.0);
  FlatMsgView<ViewFoo> view;
  ASSERT_TRUE(!view.template Match(variant_list));
}

// clang-format off
FLAT_MSG_VIEW_BEGIN(RepeatedFoo);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int32_t, int32_value);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int16_t, int16_value);
  FLAT_MSG_VIEW_DEFINE_PATTERN(float, float_value);
FLAT_MSG_VIEW_END(RepeatedFoo);
// clang-format on

TEST(FlatMsgView, repeated_empty) {
  std::vector<FlatMsg<VariantFoo>> variant_list(2);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_float_value(50.0);
  FlatMsgView<RepeatedFoo> view;
  ASSERT_TRUE(view.Match(variant_list));
  ASSERT_EQ(view->int16_value_size(), 0);
}

TEST(FlatMsgView, repeated_empty_failed) {
  std::vector<FlatMsg<VariantFoo>> variant_list(2);
  variant_list.at(0)->set_float_value(50.0);
  variant_list.at(1)->set_int32_value(40);
  FlatMsgView<RepeatedFoo> view;
  ASSERT_TRUE(!view.Match(variant_list));
}

TEST(FlatMsgView, repeated_one) {
  std::vector<FlatMsg<VariantFoo>> variant_list(3);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_int16_value(45);
  variant_list.at(2)->set_float_value(50.0);
  FlatMsgView<RepeatedFoo> view;
  ASSERT_TRUE(view.Match(variant_list));
  ASSERT_EQ(view->int16_value_size(), 1);
  ASSERT_EQ(view->int16_value(0), 45);
}

TEST(FlatMsgView, repeated_one_failed) {
  std::vector<FlatMsg<VariantFoo>> variant_list(3);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_float_value(50.0);
  variant_list.at(2)->set_int16_value(45);
  FlatMsgView<RepeatedFoo> view;
  ASSERT_TRUE(!view.Match(variant_list));
}

TEST(FlatMsgView, repeated_many) {
  std::vector<FlatMsg<VariantFoo>> variant_list(4);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_int16_value(45);
  variant_list.at(2)->set_int16_value(45);
  variant_list.at(3)->set_float_value(50.0);
  FlatMsgView<RepeatedFoo> view;
  ASSERT_TRUE(view.Match(variant_list));
  ASSERT_EQ(view->int16_value_size(), 2);
  ASSERT_EQ(view->int16_value(0), 45);
  ASSERT_EQ(view->int16_value(1), 45);
}

TEST(FlatMsgView, repeated_many_failed) {
  std::vector<FlatMsg<VariantFoo>> variant_list(4);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_int16_value(45);
  variant_list.at(2)->set_float_value(45.0);
  variant_list.at(3)->set_float_value(50.0);
  FlatMsgView<RepeatedFoo> view;
  ASSERT_TRUE(!view.Match(variant_list));
}

// clang-format off
FLAT_MSG_VIEW_BEGIN(LastFieldRepeatedFoo);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int32_t, int32_value);
  FLAT_MSG_VIEW_DEFINE_PATTERN(float, float_value);
  FLAT_MSG_VIEW_DEFINE_REPEATED_PATTERN(int16_t, int16_value);
FLAT_MSG_VIEW_END(LastFieldRepeatedFoo);
// clang-format on

TEST(FlatMsgView, last_field_repeated_empty) {
  std::vector<FlatMsg<VariantFoo>> variant_list(2);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_float_value(50.0);
  FlatMsgView<LastFieldRepeatedFoo> view;
  ASSERT_TRUE(view.Match(variant_list));
  ASSERT_EQ(view->int16_value_size(), 0);
}

TEST(FlatMsgView, last_field_repeated_empty_failed) {
  std::vector<FlatMsg<VariantFoo>> variant_list(2);
  variant_list.at(0)->set_float_value(50.0);
  variant_list.at(1)->set_int32_value(40);
  FlatMsgView<LastFieldRepeatedFoo> view;
  ASSERT_TRUE(!view.Match(variant_list));
}

TEST(FlatMsgView, last_field_repeated_one) {
  std::vector<FlatMsg<VariantFoo>> variant_list(3);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_float_value(50.0);
  variant_list.at(2)->set_int16_value(45);
  FlatMsgView<LastFieldRepeatedFoo> view;
  ASSERT_TRUE(view.Match(variant_list));
  ASSERT_EQ(view->int16_value_size(), 1);
  ASSERT_EQ(view->int16_value(0), 45);
}

TEST(FlatMsgView, last_field_repeated_one_failed) {
  std::vector<FlatMsg<VariantFoo>> variant_list(3);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_int16_value(45);
  variant_list.at(2)->set_float_value(50.0);
  FlatMsgView<LastFieldRepeatedFoo> view;
  ASSERT_TRUE(!view.Match(variant_list));
}

TEST(FlatMsgView, last_field_repeated_many) {
  std::vector<FlatMsg<VariantFoo>> variant_list(4);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_float_value(50.0);
  variant_list.at(2)->set_int16_value(45);
  variant_list.at(3)->set_int16_value(45);
  FlatMsgView<LastFieldRepeatedFoo> view;
  ASSERT_TRUE(view.Match(variant_list));
  ASSERT_EQ(view->int16_value_size(), 2);
  ASSERT_EQ(view->int16_value(0), 45);
  ASSERT_EQ(view->int16_value(1), 45);
}

TEST(FlatMsgView, last_field_repeated_many_failed) {
  std::vector<FlatMsg<VariantFoo>> variant_list(4);
  variant_list.at(0)->set_int32_value(40);
  variant_list.at(1)->set_int16_value(45);
  variant_list.at(2)->set_float_value(50.0);
  variant_list.at(3)->set_int16_value(45);
  FlatMsgView<LastFieldRepeatedFoo> view;
  ASSERT_TRUE(!view.Match(variant_list));
}

}  // namespace

}  // namespace test

}  // namespace oneflow
