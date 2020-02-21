#include <sstream>
#include <iostream>
#include "oneflow/core/common/dss.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/object_msg_reflection.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

#define DSS_DEFINE_TEST_UNION_FIELD(field_counter)                   \
  DSS_DEFINE_FIELD(field_counter, "demo dss", union_field);          \
  DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, union_case,          \
                                 OF_PP_MAKE_TUPLE_SEQ(int32_t, x, 1) \
                                     OF_PP_MAKE_TUPLE_SEQ(int64_t, y, 2));

struct TestDssUnion {
  BEGIN_DSS(DSS_GET_FIELD_COUNTER(), TestDssUnion, 0);

 public:
  struct {
    int32_t union_case;
    union {
      int32_t x;
      int64_t y;
    };
  } union_field;

  DSS_DEFINE_TEST_UNION_FIELD(DSS_GET_FIELD_COUNTER());
  END_DSS(DSS_GET_FIELD_COUNTER(), "demo dss", TestDssUnion);
};

TEST(ObjectMsgReflection, ReflectObjectMsgFields) {
  ObjectMsgFieldList obj_msg_field_list;
  ObjectMsgReflection<TestDssUnion>().ReflectObjectMsgFields(&obj_msg_field_list);
  ASSERT_EQ(obj_msg_field_list.object_msg_field().size(), 1);
  ASSERT_TRUE(obj_msg_field_list.object_msg_field(0).has_union_field_list());
  const auto& union_field_list = obj_msg_field_list.object_msg_field(0).union_field_list();
  ASSERT_EQ(union_field_list.union_name(), "union_field");
  ASSERT_EQ(union_field_list.union_field(0).field_name(), "x");
  ASSERT_EQ(union_field_list.union_field(1).field_name(), "y");
}

// clang-format off
BEGIN_OBJECT_MSG(Foo);
  OBJECT_MSG_DEFINE_OPTIONAL(int, x);
END_OBJECT_MSG(Foo);

BEGIN_OBJECT_MSG(Bar);
  OBJECT_MSG_DEFINE_OPTIONAL(int, x);
END_OBJECT_MSG(Bar);

BEGIN_OBJECT_MSG(FooListItem);
  OBJECT_MSG_DEFINE_LIST_LINK(link);
END_OBJECT_MSG(FooListItem);

BEGIN_OBJECT_MSG(FooBar);
  OBJECT_MSG_DEFINE_OPTIONAL(Foo, foo);
  OBJECT_MSG_DEFINE_OPTIONAL(Bar, bar);
  OBJECT_MSG_DEFINE_ONEOF(type,
    OBJECT_MSG_ONEOF_FIELD(Foo, oneof_foo)
    OBJECT_MSG_ONEOF_FIELD(Bar, oneof_bar));
  OBJECT_MSG_DEFINE_LIST_HEAD(FooListItem, link, foo_list);
END_OBJECT_MSG(FooBar);
// clang-format on

TEST(ObjectMsgReflection, RecursivelyReflectObjectMsgFields) {
  std::unordered_map<std::string, ObjectMsgFieldList> name2field_list;
  ObjectMsgReflection<FooBar>().RecursivelyReflectObjectMsgFields(&name2field_list);
  ASSERT_EQ(name2field_list.size(), 4);
  ASSERT_TRUE(name2field_list.find(typeid(FooBar).name()) != name2field_list.end());
  ASSERT_TRUE(name2field_list.find(typeid(Foo).name()) != name2field_list.end());
  ASSERT_TRUE(name2field_list.find(typeid(Bar).name()) != name2field_list.end());
  ASSERT_TRUE(name2field_list.find(typeid(FooListItem).name()) != name2field_list.end());
}

TEST(ObjectMsgFieldListUtil, ToDot) {
  //  std::cout << std::endl;
  //  std::cout << ObjectMsgListReflection<FooBar>().ToDot() << std::endl;
  //  std::cout << std::endl;
}
}  // namespace

}  // namespace test

}  // namespace oneflow
