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
// include sstream first to avoid some compiling error
// caused by the following trick
// reference: https://gcc.gnu.org/bugzilla/show_bug.cgi?id=65899
#include <sstream>
#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/object_msg/flat_msg.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace intrusive {

namespace test {

namespace {

TEST(Ref, ref_cnt) {
  class Foo final : public Ref {
   public:
    Foo() = default;
  };
  Foo foo;
  foo.InitRefCount();
  foo.IncreaseRefCount();
  foo.IncreaseRefCount();
  ASSERT_EQ(foo.DecreaseRefCount(), 1);
  ASSERT_EQ(foo.DecreaseRefCount(), 0);
}

// clang-format off
OBJECT_MSG_BEGIN(IntrusiveFoo)
 public:
  void __Init__() { clear_is_deleted(); }

  // Getters
  int8_t x() const { return x_; }
  int32_t foo() const { return foo_; }
  int16_t bar() const { return bar_; }
  int64_t foobar() const { return foobar_; }
  bool has_is_deleted() const { return is_deleted_ != nullptr; }
  const std::string& is_deleted() const { return *is_deleted_; }

  // Setters
  void set_x(int8_t val) { x_ = val; }
  void set_foo(int32_t val) { foo_ = val; }
  void set_bar(int16_t val) { bar_ = val; }
  void set_foobar(int64_t val) { foobar_ = val; }
  void set_is_deleted(std::string* val) { is_deleted_ = val; }
  std::string* mut_is_deleted() { return is_deleted_; }
  std::string* mutable_is_deleted() { return is_deleted_; }
  void clear_is_deleted() { is_deleted_ = nullptr; }

  void __Delete__();

  OBJECT_MSG_DEFINE_FIELD(int8_t, x_);
  OBJECT_MSG_DEFINE_FIELD(int32_t, foo_);
  OBJECT_MSG_DEFINE_FIELD(int16_t, bar_);
  OBJECT_MSG_DEFINE_FIELD(int64_t, foobar_);
  OBJECT_MSG_DEFINE_FIELD(std::string*, is_deleted_);

OBJECT_MSG_END(IntrusiveFoo)
// clang-format on

void IntrusiveFoo::__Delete__() {
  if (mutable_is_deleted()) { *mutable_is_deleted() = "deleted"; }
}

TEST(OBJECT_MSG, naive) {
  auto foo = intrusive::MakeShared<IntrusiveFoo>();
  foo->set_bar(9527);
  ASSERT_TRUE(foo->bar() == 9527);
}

TEST(OBJECT_MSG, __delete__) {
  std::string is_deleted;
  {
    auto foo = intrusive::MakeShared<IntrusiveFoo>();
    foo->set_bar(9527);
    foo->set_is_deleted(&is_deleted);
    ASSERT_EQ(foo->bar(), 9527);
  }
  ASSERT_TRUE(is_deleted == "deleted");
}

// clang-format off
OBJECT_MSG_BEGIN(IntrusiveBar)
 public:
  void __Init__() { clear_is_deleted(); }

  // Getters
  const IntrusiveFoo& foo() const {
    if (foo_) { return foo_.Get(); }
    static const auto default_val = intrusive::MakeShared<IntrusiveFoo>();
    return default_val.Get();
  }
  const std::string& is_deleted() const { return *is_deleted_; }
  bool has_is_deleted() const { return is_deleted_ != nullptr; }
  // Setters
  IntrusiveFoo* mut_foo() { return mutable_foo(); }
  IntrusiveFoo* mutable_foo() {
    if (!foo_) { foo_ = intrusive::MakeShared<IntrusiveFoo>(); }
    return foo_.Mutable();
  }
  std::string* mut_is_deleted() { return is_deleted_; }
  std::string* mutable_is_deleted() { return is_deleted_; }
  void set_is_deleted(std::string* val) { is_deleted_ = val; }
  void clear_is_deleted() { is_deleted_ = nullptr; }

  OBJECT_MSG_DEFINE_FIELD(intrusive::SharedPtr<IntrusiveFoo>, foo_);
  OBJECT_MSG_DEFINE_FIELD(std::string*, is_deleted_);

 public:
  void __Delete__(){
    if (mutable_is_deleted()) { *mutable_is_deleted() = "bar_deleted"; }
  }
OBJECT_MSG_END(IntrusiveBar)
// clang-format on

TEST(OBJECT_MSG, nested_objects) {
  auto bar = intrusive::MakeShared<IntrusiveBar>();
  bar->mutable_foo()->set_bar(9527);
  ASSERT_TRUE(bar->foo().bar() == 9527);
}

TEST(OBJECT_MSG, nested_delete) {
  std::string bar_is_deleted;
  std::string is_deleted;
  {
    auto bar = intrusive::MakeShared<IntrusiveBar>();
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
FLAT_MSG_BEGIN(FlatMsgDemo)
  FLAT_MSG_DEFINE_ONEOF(type,
      FLAT_MSG_ONEOF_FIELD(int32_t, int32_field)
      FLAT_MSG_ONEOF_FIELD(float, float_field));
FLAT_MSG_END(FlatMsgDemo)
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(IntrusiveContainerDemo)
 public:
  // Getters
  const FlatMsgDemo& flat_field() const { return flat_field_.Get(); }
  // Setters
  FlatMsgDemo* mut_flat_field() { return flat_field_.Mutable(); }
  FlatMsgDemo* mutable_flat_field() { return flat_field_.Mutable(); }

  OBJECT_MSG_DEFINE_FIELD(FlatMsg<FlatMsgDemo>, flat_field_);
OBJECT_MSG_END(IntrusiveContainerDemo)
// clang-format on

TEST(OBJECT_MSG, flat_msg_field) {
  auto obj = intrusive::MakeShared<IntrusiveContainerDemo>();
  ASSERT_TRUE(!obj->flat_field().has_int32_field());
  obj->mutable_flat_field()->set_int32_field(33);
  ASSERT_TRUE(obj->flat_field().has_int32_field());
  ASSERT_EQ(obj->flat_field().int32_field(), 33);
}

// clang-format off
OBJECT_MSG_BEGIN(TestIntrusiveField);
  static_assert(OBJECT_MSG_FIELD_COUNTER == 0, "");
  static_assert(OBJECT_MSG_FIELD_COUNTER == 0, "");
  OBJECT_MSG_DEFINE_FIELD(int32_t, a);
  static_assert(OBJECT_MSG_FIELD_COUNTER == 1, "");
  static_assert(OBJECT_MSG_FIELD_COUNTER == 1, "");
  OBJECT_MSG_DEFINE_FIELD(int64_t, b);
  static_assert(OBJECT_MSG_FIELD_COUNTER == 2, "");
  static_assert(OBJECT_MSG_FIELD_COUNTER == 2, "");
  OBJECT_MSG_DEFINE_FIELD(int8_t, c);
  static_assert(OBJECT_MSG_FIELD_COUNTER == 3, "");
  static_assert(OBJECT_MSG_FIELD_COUNTER == 3, "");
  OBJECT_MSG_DEFINE_FIELD(int64_t, d);
  static_assert(OBJECT_MSG_FIELD_COUNTER == 4, "");
  static_assert(OBJECT_MSG_FIELD_COUNTER == 4, "");
OBJECT_MSG_END(TestIntrusiveField);
// clang-format on

TEST(OBJECT_MSG, object_msg_field_number) {
  static_assert(OBJECT_MSG_FIELD_NUMBER(TestIntrusiveField, a) == 1, "");
  static_assert(OBJECT_MSG_FIELD_NUMBER(TestIntrusiveField, b) == 2, "");
  static_assert(OBJECT_MSG_FIELD_NUMBER(TestIntrusiveField, c) == 3, "");
  static_assert(OBJECT_MSG_FIELD_NUMBER(TestIntrusiveField, d) == 4, "");
}

TEST(OBJECT_MSG, object_msg_field_type) {
  static_assert(std::is_same<OBJECT_MSG_FIELD_TYPE(TestIntrusiveField, 1), int32_t>::value, "");
  static_assert(std::is_same<OBJECT_MSG_FIELD_TYPE(TestIntrusiveField, 2), int64_t>::value, "");
  static_assert(std::is_same<OBJECT_MSG_FIELD_TYPE(TestIntrusiveField, 3), int8_t>::value, "");
  static_assert(std::is_same<OBJECT_MSG_FIELD_TYPE(TestIntrusiveField, 4), int64_t>::value, "");
}

TEST(OBJECT_MSG, object_msg_field_offset) {
  static_assert(OBJECT_MSG_FIELD_OFFSET(TestIntrusiveField, 1) == 0, "");
  static_assert(OBJECT_MSG_FIELD_OFFSET(TestIntrusiveField, 2) == 8, "");
  static_assert(OBJECT_MSG_FIELD_OFFSET(TestIntrusiveField, 3) == 16, "");
  static_assert(OBJECT_MSG_FIELD_OFFSET(TestIntrusiveField, 4) == 24, "");
}

}  // namespace

}  // namespace test

}  // namespace intrusive

}  // namespace oneflow
