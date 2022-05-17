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
#include "gtest/gtest.h"
#define private public
#include "oneflow/core/common/util.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/intrusive/flat_msg.h"
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

class IntrusiveFoo final : public intrusive::Base {
 public:
  void __Init__() { clear_is_deleted(); }
  void __Delete__();

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
  void clear_is_deleted() { is_deleted_ = nullptr; }

  size_t ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  IntrusiveFoo() : intrusive_ref_(), x_(), foo_(), bar_(), foobar_(), is_deleted_() {}
  intrusive::Ref intrusive_ref_;
  int8_t x_;
  int32_t foo_;
  int16_t bar_;
  int64_t foobar_;
  std::string* is_deleted_;
};

void IntrusiveFoo::__Delete__() {
  if (mut_is_deleted()) { *mut_is_deleted() = "deleted"; }
}

TEST(intrusive, naive) {
  auto foo = intrusive::make_shared<IntrusiveFoo>();
  foo->set_bar(9527);
  ASSERT_TRUE(foo->bar() == 9527);
}

TEST(intrusive, __delete__) {
  std::string is_deleted;
  {
    auto foo = intrusive::make_shared<IntrusiveFoo>();
    foo->set_bar(9527);
    foo->set_is_deleted(&is_deleted);
    ASSERT_EQ(foo->bar(), 9527);
  }
  ASSERT_TRUE(is_deleted == "deleted");
}

class IntrusiveBar final : public intrusive::Base {
 public:
  void __Init__() { clear_is_deleted(); }
  void __Delete__() {
    if (mut_is_deleted()) { *mut_is_deleted() = "bar_deleted"; }
  }

  // Getters
  const IntrusiveFoo& foo() const {
    if (foo_) { return foo_.Get(); }
    static const auto default_val = intrusive::make_shared<IntrusiveFoo>();
    return default_val.Get();
  }
  const std::string& is_deleted() const { return *is_deleted_; }
  bool has_is_deleted() const { return is_deleted_ != nullptr; }

  // Setters
  IntrusiveFoo* mut_foo() {
    if (!foo_) { foo_ = intrusive::make_shared<IntrusiveFoo>(); }
    return foo_.Mutable();
  }
  std::string* mut_is_deleted() { return is_deleted_; }
  void set_is_deleted(std::string* val) { is_deleted_ = val; }
  void clear_is_deleted() { is_deleted_ = nullptr; }

  size_t ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  IntrusiveBar() : intrusive_ref_(), foo_(), is_deleted_() {}
  intrusive::Ref intrusive_ref_;
  intrusive::shared_ptr<IntrusiveFoo> foo_;
  std::string* is_deleted_;
};

TEST(intrusive, nested_objects) {
  auto bar = intrusive::make_shared<IntrusiveBar>();
  bar->mut_foo()->set_bar(9527);
  ASSERT_TRUE(bar->foo().bar() == 9527);
}

TEST(intrusive, nested_delete) {
  std::string bar_is_deleted;
  std::string is_deleted;
  {
    auto bar = intrusive::make_shared<IntrusiveBar>();
    bar->set_is_deleted(&bar_is_deleted);
    auto* foo = bar->mut_foo();
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

class IntrusiveContainerDemo final : public intrusive::Base {
 public:
  // Getters
  const FlatMsgDemo& flat_field() const { return flat_field_.Get(); }
  // Setters
  FlatMsgDemo* mut_flat_field() { return flat_field_.Mutable(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  IntrusiveContainerDemo() : intrusive_ref_(), flat_field_() {}
  intrusive::Ref intrusive_ref_;
  FlatMsg<FlatMsgDemo> flat_field_;
};

TEST(intrusive, flat_msg_field) {
  auto obj = intrusive::make_shared<IntrusiveContainerDemo>();
  ASSERT_TRUE(!obj->flat_field().has_int32_field());
  obj->mut_flat_field()->set_int32_field(33);
  ASSERT_TRUE(obj->flat_field().has_int32_field());
  ASSERT_EQ(obj->flat_field().int32_field(), 33);
}

// clang-format off
REFLECTIVE_CLASS_BEGIN(TestIntrusiveField);
  TestIntrusiveField() = default;
  static_assert(REFLECTIVE_FIELD_COUNTER == 0, "");
  static_assert(REFLECTIVE_FIELD_COUNTER == 0, "");
  REFLECTIVE_CLASS_DEFINE_FIELD(int32_t, a);
  static_assert(REFLECTIVE_FIELD_COUNTER == 1, "");
  static_assert(REFLECTIVE_FIELD_COUNTER == 1, "");
  REFLECTIVE_CLASS_DEFINE_FIELD(int64_t, b);
  static_assert(REFLECTIVE_FIELD_COUNTER == 2, "");
  static_assert(REFLECTIVE_FIELD_COUNTER == 2, "");
  REFLECTIVE_CLASS_DEFINE_FIELD(int8_t, c);
  static_assert(REFLECTIVE_FIELD_COUNTER == 3, "");
  static_assert(REFLECTIVE_FIELD_COUNTER == 3, "");
  REFLECTIVE_CLASS_DEFINE_FIELD(int64_t, d);
  static_assert(REFLECTIVE_FIELD_COUNTER == 4, "");
  static_assert(REFLECTIVE_FIELD_COUNTER == 4, "");
REFLECTIVE_CLASS_END(TestIntrusiveField);
// clang-format on

TEST(intrusive, intrusive_field_number) {
  static_assert(REFLECTIVE_FIELD_NUMBER(TestIntrusiveField, a) == 1, "");
  static_assert(REFLECTIVE_FIELD_NUMBER(TestIntrusiveField, b) == 2, "");
  static_assert(REFLECTIVE_FIELD_NUMBER(TestIntrusiveField, c) == 3, "");
  static_assert(REFLECTIVE_FIELD_NUMBER(TestIntrusiveField, d) == 4, "");
}

TEST(intrusive, intrusive_field_type) {
  static_assert(std::is_same<REFLECTIVE_FIELD_TYPE(TestIntrusiveField, 1), int32_t>::value, "");
  static_assert(std::is_same<REFLECTIVE_FIELD_TYPE(TestIntrusiveField, 2), int64_t>::value, "");
  static_assert(std::is_same<REFLECTIVE_FIELD_TYPE(TestIntrusiveField, 3), int8_t>::value, "");
  static_assert(std::is_same<REFLECTIVE_FIELD_TYPE(TestIntrusiveField, 4), int64_t>::value, "");
}

TEST(intrusive, intrusive_field_offset) {
  static_assert(REFLECTIVE_FIELD_OFFSET(TestIntrusiveField, 1) == 0, "");
  static_assert(REFLECTIVE_FIELD_OFFSET(TestIntrusiveField, 2) == 8, "");
  static_assert(REFLECTIVE_FIELD_OFFSET(TestIntrusiveField, 3) == 16, "");
  static_assert(REFLECTIVE_FIELD_OFFSET(TestIntrusiveField, 4) == 24, "");
}

}  // namespace

}  // namespace test

}  // namespace intrusive

}  // namespace oneflow
