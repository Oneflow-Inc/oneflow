#include "oneflow/core/common/flat_msg_view.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

// clang-format off
BEGIN_FLAT_MSG(VariantFoo);
  FLAT_MSG_DEFINE_STRICT_ONEOF(_,
    FLAT_MSG_ONEOF_FIELD(int8_t, int8_value)
    FLAT_MSG_ONEOF_FIELD(int16_t, int16_value)
    FLAT_MSG_ONEOF_FIELD(int32_t, int32_value)
    FLAT_MSG_ONEOF_FIELD(float, float_value));
END_FLAT_MSG(VariantFoo);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(DefaultOneofNameVariantFoo);
  FLAT_MSG_DEFINE_STRICT_ONEOF(_,
    FLAT_MSG_ONEOF_FIELD(int8_t, int8_value)
    FLAT_MSG_ONEOF_FIELD(int16_t, int16_value)
    FLAT_MSG_ONEOF_FIELD(int32_t, int32_value)
    FLAT_MSG_ONEOF_FIELD(float, float_value));
END_FLAT_MSG(DefaultOneofNameVariantFoo);
// clang-format on

using TestOneofField =
    StructField<VariantFoo, VariantFoo::__OneofType, VariantFoo::__DssFieldOffset()>;

// clang-format off
BEGIN_FLAT_MSG(VariantList);
  FLAT_MSG_DEFINE_REPEATED(VariantFoo, foo, 16);
END_FLAT_MSG(VariantList);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG(DefaultOneofNameVariantFooList);
  FLAT_MSG_DEFINE_REPEATED(DefaultOneofNameVariantFoo, foo, 16);
END_FLAT_MSG(DefaultOneofNameVariantFooList);
// clang-format on

// clang-format off
BEGIN_FLAT_MSG_VIEW(ViewFoo);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int32_t, int32_value);
  FLAT_MSG_VIEW_DEFINE_PATTERN(int16_t, int16_value);
  FLAT_MSG_VIEW_DEFINE_PATTERN(float, float_value);
END_FLAT_MSG_VIEW(ViewFoo);
// clang-format on

TEST(FlatMsgView, match_success) {
  FlatMsg<VariantList> variant_list;
  variant_list.Mutable()->mutable_foo()->Add()->set_int32_value(30);
  variant_list.Mutable()->mutable_foo()->Add()->set_int16_value(40);
  variant_list.Mutable()->mutable_foo()->Add()->set_float_value(50.0);
  FlatMsgView<ViewFoo> view;
  ASSERT_TRUE(view->template MatchOneof<TestOneofField>(variant_list.Mutable()->mutable_foo()));
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
  ASSERT_TRUE(!view->template MatchOneof<TestOneofField>(variant_list.Mutable()->mutable_foo()));
}

TEST(FlatMsgView, init) {
  FlatMsg<DefaultOneofNameVariantFooList> variant_list;
  {
    FlatMsgView<ViewFoo> mut_view(variant_list.Mutable()->mutable_foo());
    mut_view->set_int32_value(30);
    mut_view->set_int16_value(40);
    mut_view->set_float_value(50.0);
  }
  {
    FlatMsgView<ViewFoo> view;
    ASSERT_TRUE(view->Match(variant_list.Mutable()->mutable_foo()));
    ASSERT_EQ(view->int32_value(), 30);
    ASSERT_EQ(view->int16_value(), 40);
    ASSERT_EQ(view->float_value(), 50.0);
  }
}

}  // namespace

}  // namespace test

}  // namespace oneflow
