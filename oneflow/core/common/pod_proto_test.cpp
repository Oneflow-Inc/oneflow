#include "oneflow/core/common/util.h"
#include "oneflow/core/common/pod_proto.h"

namespace oneflow {

namespace test {

TEST(POD_PROTO, naive) {
  struct Foo final {
    POD_PROTO_DEFINE_FIELD(int, x);
    POD_PROTO_DEFINE_ONEOF(type,
                           POD_PROTO_ONEOF_FIELD(int, case_0) POD_PROTO_ONEOF_FIELD(int, case_1));
  };
  Foo foo;
  foo.mutable_case_0();
  Foo::POD_PROTO_ONEOF_ENUM_TYPE(type) x = foo.type_case();
  ASSERT_TRUE(x == Foo::POD_PROTO_ONEOF_ENUM_VALUE(case_0));
  *foo.mutable_case_1() = 9527;
  ASSERT_EQ(foo.case_1(), 9527);
}

}  // namespace test

}  // namespace oneflow
