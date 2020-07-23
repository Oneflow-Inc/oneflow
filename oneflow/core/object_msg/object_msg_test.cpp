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
#include "oneflow/core/common/util.h"
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace test {

// clang-format off
OBJECT_MSG_BEGIN(ObjectMsgFoo)
  OBJECT_MSG_DEFINE_OPTIONAL(int8_t, x);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, foo);
  OBJECT_MSG_DEFINE_OPTIONAL(int16_t, bar);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, foobar);
  OBJECT_MSG_DEFINE_PTR(std::string, is_deleted);

 public:
  void __Delete__();

OBJECT_MSG_END(ObjectMsgFoo)
// clang-format on

void ObjectMsgFoo::__Delete__() {
  if (mutable_is_deleted()) { *mutable_is_deleted() = "deleted"; }
}

TEST(OBJECT_MSG, naive) {
  auto foo = ObjectMsgPtr<ObjectMsgFoo>::New();
  foo->set_bar(9527);
  ASSERT_TRUE(foo->bar() == 9527);
}

TEST(OBJECT_MSG, __delete__) {
  std::string is_deleted;
  {
    auto foo = ObjectMsgPtr<ObjectMsgFoo>::New();
    foo->set_bar(9527);
    foo->set_is_deleted(&is_deleted);
    ASSERT_EQ(foo->bar(), 9527);
  }
  ASSERT_TRUE(is_deleted == "deleted");
}

// clang-format off
OBJECT_MSG_BEGIN(ObjectMsgBar)
  OBJECT_MSG_DEFINE_OPTIONAL(ObjectMsgFoo, foo);
  OBJECT_MSG_DEFINE_PTR(std::string, is_deleted);

 public:
  void __Delete__(){
    if (mutable_is_deleted()) { *mutable_is_deleted() = "bar_deleted"; }
  }
OBJECT_MSG_END(ObjectMsgBar)
// clang-format on

TEST(OBJECT_MSG, nested_objects) {
  auto bar = ObjectMsgPtr<ObjectMsgBar>::New();
  bar->mutable_foo()->set_bar(9527);
  ASSERT_TRUE(bar->foo().bar() == 9527);
}

TEST(OBJECT_MSG, nested_delete) {
  std::string bar_is_deleted;
  std::string is_deleted;
  {
    auto bar = ObjectMsgPtr<ObjectMsgBar>::New();
    bar->set_is_deleted(&bar_is_deleted);
    auto* foo = bar->mutable_foo();
    foo->set_bar(9527);
    foo->set_is_deleted(&is_deleted);
    ASSERT_EQ(foo->bar(), 9527);
    ASSERT_EQ(bar->ref_cnt(), 1);
    ASSERT_EQ(foo->ref_cnt(), 1);
  }
  ASSERT_EQ(is_deleted, std::string("deleted"));
  ASSERT_EQ(bar_is_deleted, std::string("bar_deleted"));
}

// clang-format off
OBJECT_MSG_BEGIN(TestScalarOneof)
  OBJECT_MSG_DEFINE_ONEOF(type,
      OBJECT_MSG_ONEOF_FIELD(int32_t, x)
      OBJECT_MSG_ONEOF_FIELD(int64_t, foo));
OBJECT_MSG_END(TestScalarOneof)
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(TestPtrOneof)
  OBJECT_MSG_DEFINE_ONEOF(type,
      OBJECT_MSG_ONEOF_FIELD(ObjectMsgFoo, foo)
      OBJECT_MSG_ONEOF_FIELD(int32_t, int_field));
OBJECT_MSG_END(TestPtrOneof)
// clang-format on

TEST(OBJECT_MSG, oneof_get) {
  auto test_oneof = ObjectMsgPtr<TestPtrOneof>::New();
  auto& obj = *test_oneof;
  const auto* default_foo_ptr = &obj.foo();
  ASSERT_EQ(obj.foo().x(), 0);
  ASSERT_TRUE(!obj.has_foo());
  obj.mutable_foo();
  ASSERT_TRUE(obj.has_foo());
  ASSERT_EQ(obj.foo().x(), 0);
  ASSERT_NE(default_foo_ptr, &obj.foo());
};

TEST(OBJECT_MSG, oneof_release) {
  auto test_oneof = ObjectMsgPtr<TestPtrOneof>::New();
  auto& obj = *test_oneof;
  const auto* default_foo_ptr = &obj.foo();
  ASSERT_EQ(obj.foo().x(), 0);
  obj.mutable_foo();
  ASSERT_EQ(obj.foo().x(), 0);
  ASSERT_NE(default_foo_ptr, &obj.foo());
  {
    std::string is_delete;
    obj.mutable_foo()->set_is_deleted(&is_delete);
    obj.mutable_int_field();
    ASSERT_EQ(is_delete, "deleted");
  }
  {
    std::string is_delete;
    obj.mutable_foo()->set_is_deleted(&is_delete);
    obj.mutable_int_field();
    ASSERT_EQ(is_delete, "deleted");
  }
};

TEST(OBJECT_MSG, oneof_reset) {
  auto test_oneof = ObjectMsgPtr<TestPtrOneof>::New();
  auto& obj = *test_oneof;
  const auto* default_foo_ptr = &obj.foo();
  ASSERT_EQ(obj.foo().x(), 0);
  obj.mutable_foo();
  ASSERT_EQ(obj.foo().x(), 0);
  ASSERT_NE(default_foo_ptr, &obj.foo());
  {
    auto another_oneof = ObjectMsgPtr<TestPtrOneof>::New();
    std::string is_delete;
    obj.mutable_foo()->set_is_deleted(&is_delete);
    another_oneof->reset_foo(obj.mut_foo());
    obj.mutable_int_field();
    ASSERT_EQ(is_delete, "");
    another_oneof->mutable_int_field();
    ASSERT_EQ(is_delete, "deleted");
  }
  {
    std::string is_delete;
    obj.mutable_foo()->set_is_deleted(&is_delete);
    obj.mutable_int_field();
    ASSERT_EQ(is_delete, "deleted");
  }
};

TEST(OBJECT_MSG, oneof_clear) {
  auto test_oneof = ObjectMsgPtr<TestPtrOneof>::New();
  auto& obj = *test_oneof;
  const auto* default_foo_ptr = &obj.foo();
  ASSERT_EQ(obj.foo().x(), 0);
  obj.mutable_foo();
  ASSERT_EQ(obj.foo().x(), 0);
  ASSERT_NE(default_foo_ptr, &obj.foo());
  {
    std::string is_delete;
    obj.mutable_foo()->set_is_deleted(&is_delete);
    ASSERT_TRUE(!obj.has_int_field());
    obj.clear_int_field();
    ASSERT_TRUE(!obj.has_int_field());
    ASSERT_TRUE(obj.has_foo());
    obj.clear_foo();
    ASSERT_TRUE(!obj.has_foo());
    ASSERT_EQ(is_delete, "deleted");
  }
};

TEST(OBJECT_MSG, oneof_set) {
  auto test_oneof = ObjectMsgPtr<TestPtrOneof>::New();
  auto& obj = *test_oneof;
  const auto* default_foo_ptr = &obj.foo();
  ASSERT_EQ(obj.foo().x(), 0);
  obj.mutable_foo();
  ASSERT_EQ(obj.foo().x(), 0);
  ASSERT_NE(default_foo_ptr, &obj.foo());
  {
    std::string is_delete;
    obj.mutable_foo()->set_is_deleted(&is_delete);
    ASSERT_TRUE(!obj.has_int_field());
    obj.clear_int_field();
    ASSERT_TRUE(!obj.has_int_field());
    ASSERT_TRUE(obj.has_foo());
    obj.set_int_field(30);
    ASSERT_TRUE(!obj.has_foo());
    ASSERT_EQ(is_delete, "deleted");
  }
};

// clang-format off
FLAT_MSG_BEGIN(FlatMsgDemo)
  FLAT_MSG_DEFINE_ONEOF(type,
      FLAT_MSG_ONEOF_FIELD(int32_t, int32_field)
      FLAT_MSG_ONEOF_FIELD(float, float_field));
FLAT_MSG_END(FlatMsgDemo)
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(ObjectMsgContainerDemo)
  OBJECT_MSG_DEFINE_FLAT_MSG(FlatMsgDemo, flat_field);
OBJECT_MSG_END(ObjectMsgContainerDemo)
// clang-format on

TEST(OBJECT_MSG, flat_msg_field) {
  auto obj = ObjectMsgPtr<ObjectMsgContainerDemo>::New();
  ASSERT_TRUE(obj->has_flat_field());
  ASSERT_TRUE(!obj->flat_field().has_int32_field());
  obj->mutable_flat_field()->set_int32_field(33);
  ASSERT_TRUE(obj->flat_field().has_int32_field());
  ASSERT_EQ(obj->flat_field().int32_field(), 33);
}

}  // namespace test

}  // namespace oneflow
