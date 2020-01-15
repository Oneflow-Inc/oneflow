#include "oneflow/core/common/util.h"
#include "oneflow/core/common/flat_msg.h"

namespace oneflow {

struct TestOptional final {
  FLAT_MSG(TestOptional);

  FLAT_MSG_DEFINE_FIELD(int, bar);
};

TEST(FlatMsg, optional) {
  FlatMsg<TestOptional> foo_box;
  auto& foo = *foo_box.Mutable();
  ASSERT_TRUE(!foo.has_bar());
  ASSERT_TRUE(!foo.has_bar());
  *foo.mutable_bar() = 9527;
  ASSERT_TRUE(foo.has_bar());
  ASSERT_EQ(foo.bar(), 9527);
}

struct FooOneof final {
  FLAT_MSG(FooOneof);

  FLAT_MSG_DEFINE_FIELD(int, x);
  // clang-format off
  FLAT_MSG_DEFINE_ONEOF(type,
      FLAT_MSG_ONEOF_FIELD(int, case_0)
      FLAT_MSG_ONEOF_FIELD(int, case_1));
  // clang-format on
};

TEST(FlatMsg, oneof) {
  FlatMsg<FooOneof> foo_box;
  auto& foo = *foo_box.Mutable();
  foo.mutable_case_0();
  ASSERT_TRUE(foo.has_case_0());
  FooOneof::FLAT_MSG_ONEOF_ENUM_TYPE(type) x = foo.type_case();
  ASSERT_TRUE(x == FooOneof::FLAT_MSG_ONEOF_ENUM_VALUE(case_0));
  *foo.mutable_case_1() = 9527;
  ASSERT_TRUE(foo.has_case_1());
  ASSERT_EQ(foo.case_1(), 9527);
}

struct FooRepeated final {
  FLAT_MSG(FooRepeated);

  FLAT_MSG_DEFINE_REPEATED_FIELD(TestOptional, bar, 10);
};

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

}  // namespace oneflow
