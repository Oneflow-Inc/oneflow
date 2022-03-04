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

namespace oneflow {

namespace test {

namespace {

class TestListItem : public intrusive::Base {
 public:
  void __Init__() { clear_cnt(); }
  void __Delete__() {
    if (has_cnt()) { --*mut_cnt(); }
  }

  // Getters
  bool has_cnt() const { return cnt_ != nullptr; }
  int cnt() const { return *cnt_; }
  bool is_foo_list_empty() const { return foo_list_.empty(); }

  // Setters
  void set_cnt(int* val) { cnt_ = val; }
  void clear_cnt() { cnt_ = nullptr; }
  int* mut_cnt() { return cnt_; }

  size_t ref_cnt() const { return intrusive_ref_.ref_cnt(); }

  intrusive::ListHook foo_list_;

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  TestListItem() : foo_list_(), intrusive_ref_(), cnt_() {}
  intrusive::Ref intrusive_ref_;
  int* cnt_;
};

using TestList = intrusive::List<INTRUSIVE_FIELD(TestListItem, foo_list_)>;

TEST(List, empty) {
  TestList foo_list;
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(foo_list.size(), 0);
}

TEST(List, empty_Begin) {
  TestList foo_list;
  intrusive::shared_ptr<TestListItem> obj_ptr;
  obj_ptr = foo_list.Begin();
  ASSERT_TRUE(!obj_ptr);
  intrusive::shared_ptr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(!obj_ptr);
}

TEST(List, empty_Next) {
  TestList foo_list;
  intrusive::shared_ptr<TestListItem> obj_ptr;
  intrusive::shared_ptr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(!obj_ptr);
  ASSERT_TRUE(!next);
  obj_ptr = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(!obj_ptr);
  obj_ptr = next;
  next = foo_list.Next(next.Mutable());
  ASSERT_TRUE(!obj_ptr);
  ASSERT_TRUE(!next);
}

TEST(List, PushFront) {
  TestList foo_list;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushFront(item0.Mutable());
  foo_list.PushFront(item1.Mutable());
  intrusive::shared_ptr<TestListItem> obj_ptr;
  intrusive::shared_ptr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item1);
  ASSERT_TRUE(next == item0);
}

TEST(List, destructor) {
  int elem_cnt = 2;
  {
    TestList foo_list;
    auto item0 = intrusive::make_shared<TestListItem>();
    item0->set_cnt(&elem_cnt);
    auto item1 = intrusive::make_shared<TestListItem>();
    item1->set_cnt(&elem_cnt);
    foo_list.PushFront(item0.Mutable());
    foo_list.PushFront(item1.Mutable());
  }
  ASSERT_EQ(elem_cnt, 0);
  elem_cnt = 2;
  auto item0 = intrusive::make_shared<TestListItem>();
  {
    TestList foo_list;
    item0->set_cnt(&elem_cnt);
    auto item1 = intrusive::make_shared<TestListItem>();
    item1->set_cnt(&elem_cnt);
    foo_list.PushFront(item0.Mutable());
    foo_list.PushFront(item1.Mutable());
  }
  ASSERT_EQ(elem_cnt, 1);
}

TEST(List, PushBack) {
  TestList foo_list;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  intrusive::shared_ptr<TestListItem> obj_ptr;
  intrusive::shared_ptr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(next == item1);
}

TEST(List, Erase) {
  TestList foo_list;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.Erase(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 1);
  intrusive::shared_ptr<TestListItem> obj_ptr;
  intrusive::shared_ptr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(!next);
}

TEST(List, PopBack) {
  TestList foo_list;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.PopBack();
  ASSERT_EQ(item1->ref_cnt(), 1);
  intrusive::shared_ptr<TestListItem> obj_ptr;
  intrusive::shared_ptr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(!next);
}

TEST(List, PopFront) {
  TestList foo_list;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  foo_list.PopFront();
  ASSERT_EQ(item0->ref_cnt(), 1);
  intrusive::shared_ptr<TestListItem> obj_ptr;
  intrusive::shared_ptr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(!next);
}

TEST(List, Clear) {
  TestList foo_list;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.Clear();
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(item1->ref_cnt(), 1);
}

TEST(List, UNSAFE_FOR_EACH_PTR) {
  TestList foo_list;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  int i = 0;
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(item, &foo_list) {
    if (i == 0) {
      ASSERT_TRUE(item == item0.Mutable());
    } else if (i == 1) {
      ASSERT_TRUE(item == item1.Mutable());
    }
    ++i;
  }
  ASSERT_EQ(i, 2);
}

TEST(List, FOR_EACH) {
  TestList foo_list;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  int i = 0;
  INTRUSIVE_FOR_EACH(item, &foo_list) {
    if (i == 0) {
      ASSERT_TRUE(item == item0);
      foo_list.Erase(item.Mutable());
    } else if (i == 1) {
      ASSERT_TRUE(item == item1);
      foo_list.Erase(item.Mutable());
    }
    ++i;
  }
  ASSERT_EQ(i, 2);
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(item1->ref_cnt(), 1);
}

class TestIntrusiveListHead final : public intrusive::Base {
 public:
  // types
  using FooList = intrusive::List<INTRUSIVE_FIELD(TestListItem, foo_list_)>;
  // Getters
  const FooList& foo_list() const { return foo_list_; }
  // Setters
  FooList* mut_foo_list() { return &foo_list_; }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  TestIntrusiveListHead() : intrusive_ref_(), foo_list_() {}
  intrusive::Ref intrusive_ref_;
  FooList foo_list_;
};

TEST(List, intrusive_list_for_each) {
  auto foo_list_head = intrusive::make_shared<TestIntrusiveListHead>();
  auto& foo_list = *foo_list_head->mut_foo_list();
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  int i = 0;
  INTRUSIVE_FOR_EACH(item, &foo_list) {
    if (i == 0) {
      ASSERT_TRUE(item == item0);
      foo_list.Erase(item.Mutable());
    } else if (i == 1) {
      ASSERT_TRUE(item == item1);
      foo_list.Erase(item.Mutable());
    }
    ++i;
  }
  ASSERT_EQ(i, 2);
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(item1->ref_cnt(), 1);
}

class TestIntrusiveListHeadWrapper final : public intrusive::Base {
 public:
  // Getters
  const TestIntrusiveListHead& head() const {
    if (head_) { return head_.Get(); }
    static const auto default_val = intrusive::make_shared<TestIntrusiveListHead>();
    return default_val.Get();
  }
  // Setters
  TestIntrusiveListHead* mut_head() {
    if (!head_) { head_ = intrusive::make_shared<TestIntrusiveListHead>(); }
    return head_.Mutable();
  }
  void clear_head() {
    if (head_) { head_.Reset(); }
  }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  TestIntrusiveListHeadWrapper() : intrusive_ref_(), head_() {}
  intrusive::Ref intrusive_ref_;
  intrusive::shared_ptr<TestIntrusiveListHead> head_;
};

TEST(List, nested_list_delete) {
  auto foo_list_head = intrusive::make_shared<TestIntrusiveListHeadWrapper>();
  auto& foo_list = *foo_list_head->mut_head()->mut_foo_list();
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  int i = 0;
  INTRUSIVE_UNSAFE_FOR_EACH_PTR(item, &foo_list) {
    if (i == 0) {
      ASSERT_TRUE(item == item0.Mutable());
    } else if (i == 1) {
      ASSERT_TRUE(item == item1.Mutable());
    }
    ++i;
  }
  ASSERT_EQ(i, 2);
  foo_list_head->clear_head();
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(item1->ref_cnt(), 1);
}

TEST(List, MoveTo) {
  TestList foo_list;
  TestList foo_list0;
  auto item0 = intrusive::make_shared<TestListItem>();
  auto item1 = intrusive::make_shared<TestListItem>();
  ASSERT_EQ(item0->is_foo_list_empty(), true);
  ASSERT_EQ(item1->is_foo_list_empty(), true);
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->is_foo_list_empty(), false);
  ASSERT_EQ(item1->is_foo_list_empty(), false);
  ASSERT_EQ(foo_list.size(), 2);
  ASSERT_EQ(foo_list0.empty(), true);
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.MoveTo(&foo_list0);
  ASSERT_EQ(foo_list0.size(), 2);
  ASSERT_EQ(foo_list.empty(), true);
  ASSERT_TRUE(foo_list0.Begin() == item0.Mutable());
  ASSERT_TRUE(foo_list0.Last() == item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
