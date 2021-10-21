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
#include <sstream>
#include <iostream>
#include "oneflow/core/object_msg/dss.h"
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/object_msg/object_msg_reflection.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

#define DSS_DEFINE_TEST_UNION_FIELD(field_counter)                      \
  DSS_DEFINE_FIELD(field_counter, "demo dss", UnionField, union_field); \
  DSS_DEFINE_UNION_FIELD_VISITOR(field_counter, union_case,             \
                                 OF_PP_MAKE_TUPLE_SEQ(int32_t, x, 1)    \
                                     OF_PP_MAKE_TUPLE_SEQ(int64_t, y, 2));

struct TestDssUnion {
  DSS_BEGIN(DSS_GET_FIELD_COUNTER(), TestDssUnion);

 public:
  struct UnionField {
    int32_t union_case;
    union {
      int32_t x;
      int64_t y;
    };
  } union_field;

  DSS_DEFINE_TEST_UNION_FIELD(DSS_GET_FIELD_COUNTER());
  DSS_END(DSS_GET_FIELD_COUNTER(), "demo dss", TestDssUnion);
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
OBJECT_MSG_BEGIN(Foo);
  OBJECT_MSG_DEFINE_OPTIONAL(int, x);
OBJECT_MSG_END(Foo);

OBJECT_MSG_BEGIN(Bar);
  OBJECT_MSG_DEFINE_OPTIONAL(int, x);
OBJECT_MSG_END(Bar);

OBJECT_MSG_BEGIN(FooListItem);
  OBJECT_MSG_DEFINE_LIST_LINK(link);
OBJECT_MSG_END(FooListItem);

OBJECT_MSG_BEGIN(FooBar);
  OBJECT_MSG_DEFINE_OPTIONAL(Foo, foo);
  OBJECT_MSG_DEFINE_OPTIONAL(Bar, bar);
  OBJECT_MSG_DEFINE_ONEOF(type,
    OBJECT_MSG_ONEOF_FIELD(Foo, oneof_foo)
    OBJECT_MSG_ONEOF_FIELD(Bar, oneof_bar));
  OBJECT_MSG_DEFINE_LIST_HEAD(FooListItem, link, foo_list);
OBJECT_MSG_END(FooBar);
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
