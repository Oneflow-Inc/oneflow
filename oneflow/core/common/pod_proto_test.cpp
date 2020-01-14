#include "oneflow/core/common/util.h"
#include "oneflow/core/common/pod_proto.h"

namespace oneflow {

TEST(POD_PROTO, optional) {
  struct Foo final {
    POD_PROTO_DEFINE_BASIC_METHODS(Foo);
    POD_PROTO_DEFINE_FIELD(int, bar);
  };
  Foo foo;
  ASSERT_TRUE(!foo.has_case_0());
  Foo::POD_PROTO_ONEOF_ENUM_TYPE(type) x = foo.type_case();
  ASSERT_TYPE(x, Foo::POD_PROTO_ONEOF_ENUM_VALUE(type))
  *foo.mutable_case_1() = 9527;
  ASSERT_TRUE(foo.has_case_1());
  ASSERT_EQ(foo.case_1(), 9527)
}

TEST(POD_PROTO, naive) {
  struct Foo final {
    POD_PROTO_DEFINE_BASIC_METHODS(Foo);
    POD_PROTO_DEFINE_FIELD(int, x);
  // clang-format off
    POD_PROTO_DEFINE_ONEOF(type,
        POD_PROTO_ONEOF_FIELD(int, case_0)
        POD_PROTO_ONEOF_FIELD(int, case_1));
  // clang-format on
  };
  Foo foo;
  foo.mutable_case_0();
  ASSERT_TRUE(foo.has_case_0());
  Foo::POD_PROTO_ONEOF_ENUM_TYPE(type) x = foo.type_case();
  ASSERT_TYPE(x, Foo::POD_PROTO_ONEOF_ENUM_VALUE(type))
  *foo.mutable_case_1() = 9527;
  ASSERT_TRUE(foo.has_case_1());
  ASSERT_EQ(foo.case_1(), 9527)
}

}  // namespace oneflow
