#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

// clang-format off
BEGIN_FLAT_MSG(TestOptional)
  FLAT_MSG_DEFINE_FIELD(int32_t, bar);
END_FLAT_MSG(TestOptional)
// clang-format on

TEST(FlatMsg, optional) {
  FLAT_MSG(TestOptional) foo_box;
  auto& foo = *foo_box.Mutable();
  ASSERT_TRUE(!foo.has_bar());
  ASSERT_TRUE(!foo.has_bar());
  *foo.mutable_bar() = 9527;
  ASSERT_TRUE(foo.has_bar());
  ASSERT_EQ(foo.bar(), 9527);
}

// clang-format off
BEGIN_FLAT_MSG(FooOneof)
  FLAT_MSG_DEFINE_FIELD(int32_t, x);
  FLAT_MSG_DEFINE_ONEOF(type,
      FLAT_MSG_ONEOF_FIELD(int32_t, case_0)
      FLAT_MSG_ONEOF_FIELD(int32_t, case_1));
END_FLAT_MSG(FooOneof)
// clang-format on

TEST(FlatMsg, oneof) {
  FLAT_MSG(FooOneof) foo_box;
  auto& foo = *foo_box.Mutable();
  foo.mutable_case_0();
  ASSERT_TRUE(foo.has_case_0());
  FLAT_MSG_ONEOF_ENUM_TYPE(FooOneof, type) x = foo.type_case();
  ASSERT_TRUE(x == FLAT_MSG_ONEOF_ENUM_VALUE(FooOneof, case_0));
  *foo.mutable_case_1() = 9527;
  ASSERT_TRUE(foo.has_case_1());
  ASSERT_EQ(foo.case_1(), 9527);
}

// clang-format off
BEGIN_FLAT_MSG(FooRepeated)
  FLAT_MSG_DEFINE_REPEATED_FIELD(TestOptional, bar, 10);
  FLAT_MSG_DEFINE_REPEATED_FIELD(TestOptional, foobar, 10);
END_FLAT_MSG(FooRepeated)
// clang-format on

TEST(FlatMsg, repeated) {
  FLAT_MSG(FooRepeated) foo_box;
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

}  // namespace oneflow
