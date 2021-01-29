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
#include "oneflow/core/object_msg/embedded_list_iterator.h"

namespace oneflow {

namespace test {

namespace {

// clang-format off
OBJECT_MSG_BEGIN(TestListItem)
  OBJECT_MSG_DEFINE_LIST_LINK(foo_list);
  OBJECT_MSG_DEFINE_PTR(int, cnt);

 public:
  void __Delete__() {
    if (has_cnt()) { --*mutable_cnt(); }
  }
OBJECT_MSG_END(TestListItem)
// clang-format on

TEST(ObjectMsgList, empty) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(foo_list.size(), 0);
}

TEST(ObjectMsgList, empty_Begin) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  ObjectMsgPtr<TestListItem> obj_ptr;
  obj_ptr = foo_list.Begin();
  ASSERT_TRUE(!obj_ptr);
  ObjectMsgPtr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(!obj_ptr);
}

TEST(ObjectMsgList, empty_use_itrator) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  ASSERT_EQ(foo_list.begin(), foo_list.end());
  ASSERT_EQ(foo_list.begin(), foo_list.cend());
  ASSERT_EQ(foo_list.cbegin(), foo_list.end());
  ASSERT_EQ(foo_list.cbegin(), foo_list.cend());
}

TEST(ObjectMsgList, empty_Next) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  ObjectMsgPtr<TestListItem> obj_ptr;
  ObjectMsgPtr<TestListItem> next;
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

TEST(ObjectMsgList, PushFront) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushFront(item0.Mutable());
  foo_list.PushFront(item1.Mutable());
  ObjectMsgPtr<TestListItem> obj_ptr;
  ObjectMsgPtr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item1);
  ASSERT_TRUE(next == item0);
}

TEST(ObjectMsgList, destructor) {
  int elem_cnt = 2;
  {
    OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
    auto item0 = ObjectMsgPtr<TestListItem>::New();
    item0->set_cnt(&elem_cnt);
    auto item1 = ObjectMsgPtr<TestListItem>::New();
    item1->set_cnt(&elem_cnt);
    foo_list.PushFront(item0.Mutable());
    foo_list.PushFront(item1.Mutable());
  }
  ASSERT_EQ(elem_cnt, 0);
  elem_cnt = 2;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  {
    OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
    item0->set_cnt(&elem_cnt);
    auto item1 = ObjectMsgPtr<TestListItem>::New();
    item1->set_cnt(&elem_cnt);
    foo_list.PushFront(item0.Mutable());
    foo_list.PushFront(item1.Mutable());
  }
  ASSERT_EQ(elem_cnt, 1);
}

TEST(ObjectMsgList, PushBack) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ObjectMsgPtr<TestListItem> obj_ptr;
  ObjectMsgPtr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(next == item1);
}

TEST(ObjectMsgList, Erase) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.Erase(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 1);
  ObjectMsgPtr<TestListItem> obj_ptr;
  ObjectMsgPtr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(!next);
}

TEST(ObjectMsgList, erase_use_iterator) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  auto itr1 = foo_list.erase(foo_list.begin());
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(foo_list.size(), 1);
  ASSERT_EQ(item1->ref_cnt(), 2);
  ObjectMsgPtr<TestListItem> obj_ptr1(itr1.operator->());
  ASSERT_TRUE(obj_ptr1 == item1);
  foo_list.erase(foo_list.cbegin());
  ASSERT_EQ(item1->ref_cnt(), 2);
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(foo_list.begin(), foo_list.cend());

  auto item2 = ObjectMsgPtr<TestListItem>::New();
  auto item3 = ObjectMsgPtr<TestListItem>::New();
  auto item4 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item2.Mutable());
  foo_list.PushBack(item3.Mutable());
  foo_list.PushBack(item4.Mutable());
  auto itr3 = foo_list.erase(++foo_list.begin());
  ASSERT_EQ(item2->ref_cnt(), 2);
  ASSERT_EQ(item4->ref_cnt(), 2);
  ASSERT_EQ(item3->ref_cnt(), 1);
  ObjectMsgPtr<TestListItem> obj_ptr3(itr3.operator->());
  ASSERT_EQ(obj_ptr3, item4);
  ASSERT_EQ(foo_list.size(), 2);
  foo_list.erase(++foo_list.cbegin());
  ASSERT_EQ(item2->ref_cnt(), 2);
  ASSERT_EQ(item4->ref_cnt(), 2);
  ASSERT_EQ(item3->ref_cnt(), 1);
  ASSERT_EQ(foo_list.size(), 1);

  auto item5 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item5.Mutable());
  auto itr2 = foo_list.erase(foo_list.begin(), ++foo_list.cbegin());
  ObjectMsgPtr<TestListItem> obj_ptr2(itr2.operator->());
  ASSERT_EQ(obj_ptr2, item5);
  ASSERT_EQ(item5->ref_cnt(), 3);
  ASSERT_EQ(foo_list.size(), 1);

  auto item6 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item6.Mutable());
  foo_list.erase(foo_list.begin(), foo_list.end());
  ASSERT_EQ(item6->ref_cnt(), 1);
  ASSERT_EQ(item5->ref_cnt(), 2);
  ASSERT_TRUE(foo_list.empty());

  auto item7 = ObjectMsgPtr<TestListItem>::New();
  auto item8 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item7.Mutable());
  foo_list.PushBack(item8.Mutable());
  foo_list.erase(foo_list.cbegin(), foo_list.cend());
  ASSERT_EQ(item7->ref_cnt(), 1);
  ASSERT_EQ(item8->ref_cnt(), 1);
  ASSERT_TRUE(foo_list.empty());

  auto item9 = ObjectMsgPtr<TestListItem>::New();
  auto item10 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item9.Mutable());
  foo_list.PushBack(item10.Mutable());
  foo_list.erase(foo_list.begin(), foo_list.cend());
  ASSERT_EQ(item9->ref_cnt(), 1);
  ASSERT_EQ(item10->ref_cnt(), 1);
  ASSERT_TRUE(foo_list.empty());
}

TEST(ObjectMsgList, front_and_back) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  ObjectMsgPtr<TestListItem> front_ptr(&foo_list.front());
  ObjectMsgPtr<TestListItem> back_ptr(&foo_list.back());
  ASSERT_EQ(front_ptr, item0);
  ASSERT_EQ(back_ptr, item0);
  ASSERT_EQ(front_ptr, back_ptr);

  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item1.Mutable());
  ObjectMsgPtr<TestListItem> front_ptr1(&foo_list.front());
  ObjectMsgPtr<TestListItem> back_ptr1(&foo_list.back());
  ASSERT_EQ(front_ptr1, item0);
  ASSERT_EQ(back_ptr1, item1);

  foo_list.erase(foo_list.begin());
  ObjectMsgPtr<TestListItem> front_ptr2(&foo_list.front());
  ObjectMsgPtr<TestListItem> back_ptr2(&foo_list.back());
  ASSERT_EQ(front_ptr2, item1);
  ASSERT_EQ(back_ptr2, item1);
  ASSERT_EQ(front_ptr2, back_ptr2);
}

TEST(ObjectMsgList, PopBack) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.PopBack();
  ASSERT_EQ(item1->ref_cnt(), 1);
  ObjectMsgPtr<TestListItem> obj_ptr;
  ObjectMsgPtr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(!next);
}

TEST(ObjectMsgList, PopFront) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  foo_list.PopFront();
  ASSERT_EQ(item0->ref_cnt(), 1);
  ObjectMsgPtr<TestListItem> obj_ptr;
  ObjectMsgPtr<TestListItem> next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(!next);
}

TEST(ObjectMsgList, insert_use_iterator) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  auto item2 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  auto itr1 = foo_list.insert(foo_list.begin(), *item1.Mutable());
  auto itr2 = foo_list.insert(foo_list.end(), *item2.Mutable());
  ASSERT_TRUE(foo_list.begin() == itr1);
  ASSERT_TRUE(--foo_list.end() == itr2);
  ASSERT_EQ(foo_list.size(), 3);
  {
    ObjectMsgPtr<TestListItem> first_ptr(foo_list.begin().operator->());
    ObjectMsgPtr<TestListItem> second_ptr((++foo_list.begin()).operator->());
    ObjectMsgPtr<TestListItem> third_ptr((--foo_list.end()).operator->());
    ASSERT_EQ(first_ptr, item1);
    ASSERT_EQ(second_ptr, item0);
    ASSERT_EQ(third_ptr, item2);
  }
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  ASSERT_EQ(item2->ref_cnt(), 2);
}

TEST(ObjectMsgList, iterator_to) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  auto itr1 = foo_list.iterator_to(*item0.Mutable());
  ASSERT_TRUE(foo_list.begin() == itr1);
  auto itr2 = foo_list.iterator_to(*item1.Mutable());
  ASSERT_TRUE(++foo_list.begin() == itr2);
}

TEST(ObjectMsgList, Clear) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.Clear();
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(item1->ref_cnt(), 1);
}

TEST(ObjectMsgList, UNSAFE_FOR_EACH_PTR) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  int i = 0;
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(&foo_list, item) {
    if (i == 0) {
      ASSERT_TRUE(item == item0.Mutable());
    } else if (i == 1) {
      ASSERT_TRUE(item == item1.Mutable());
    }
    ++i;
  }
  ASSERT_EQ(i, 2);
}

TEST(ObjectMsgList, FOR_EACH) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  int i = 0;
  OBJECT_MSG_LIST_FOR_EACH(&foo_list, item) {
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

TEST(ObjectMsgList, for_each_use_iterator) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  auto itr = foo_list.begin();
  while (itr != foo_list.end()) {
    itr = foo_list.erase(itr);
    if (itr == foo_list.end()) { break; }
  }
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(item1->ref_cnt(), 1);
  ASSERT_TRUE(foo_list.empty());
}

TEST(ObjectMsgList, const_for_each_use_iterator) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  int i = 0;
  for (const auto& elem : foo_list) {
    ObjectMsgPtr<TestListItem> item(const_cast<TestListItem*>(&elem));
    if (0 == i) {
      ASSERT_TRUE(item == item0);
    } else if (1 == i) {
      ASSERT_TRUE(item == item1);
    }
    ++i;
  }
  ASSERT_EQ(i, 2);
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  ASSERT_EQ(foo_list.size(), 2);
}

// clang-format off
OBJECT_MSG_BEGIN(TestObjectMsgListHead);
  OBJECT_MSG_DEFINE_LIST_HEAD(TestListItem, foo_list, foo_list);
OBJECT_MSG_END(TestObjectMsgListHead);
// clang-format on

TEST(ObjectMsg, OBJECT_MSG_DEFINE_LIST_HEAD) {
  auto foo_list_head = ObjectMsgPtr<TestObjectMsgListHead>::New();
  auto& foo_list = *foo_list_head->mutable_foo_list();
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  int i = 0;
  OBJECT_MSG_LIST_FOR_EACH(&foo_list, item) {
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

// clang-format off
OBJECT_MSG_BEGIN(TestObjectMsgListHeadWrapper);
  OBJECT_MSG_DEFINE_OPTIONAL(TestObjectMsgListHead, head);
OBJECT_MSG_END(TestObjectMsgListHeadWrapper);
// clang-format on

TEST(ObjectMsg, nested_list_delete) {
  auto foo_list_head = ObjectMsgPtr<TestObjectMsgListHeadWrapper>::New();
  auto& foo_list = *foo_list_head->mutable_head()->mutable_foo_list();
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  int i = 0;
  OBJECT_MSG_LIST_UNSAFE_FOR_EACH_PTR(&foo_list, item) {
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

TEST(ObjectMsg, MoveTo) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list0;
  auto item0 = ObjectMsgPtr<TestListItem>::New();
  auto item1 = ObjectMsgPtr<TestListItem>::New();
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

// clang-format off
OBJECT_MSG_BEGIN(SelfLoopContainer);
  // methods
  OF_PUBLIC void __Init__(bool* deleted) { set_deleted(deleted); }
  OF_PUBLIC void __Delete__() { *mut_deleted() = true; }
  // fields
  OBJECT_MSG_DEFINE_PTR(bool, deleted);
  // links
  OBJECT_MSG_DEFINE_LIST_LINK(link);
  OBJECT_MSG_DEFINE_LIST_HEAD(SelfLoopContainer, link, head);
OBJECT_MSG_END(SelfLoopContainer);
// clang-format on

TEST(ObjectMsgSelfLoopList, __Init__) {
  bool deleted = false;
  auto self_loop_head = ObjectMsgPtr<SelfLoopContainer>::New(&deleted);
  ASSERT_EQ(self_loop_head->mut_head()->container_, self_loop_head.Mutable());
}

TEST(ObjectMsgSelfLoopList, PushBack) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    ASSERT_EQ(self_loop_head0->head().size(), 1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    ASSERT_EQ(self_loop_head1->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head0->head().size(), 2);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(ObjectMsgSelfLoopList, PushFront) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
    self_loop_head0->mut_head()->PushFront(self_loop_head0.Mutable());
    ASSERT_EQ(self_loop_head0->head().size(), 1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    self_loop_head0->mut_head()->PushFront(self_loop_head1.Mutable());
    ASSERT_EQ(self_loop_head1->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head0->head().size(), 2);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(ObjectMsgSelfLoopList, EmplaceBack) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
    self_loop_head0->mut_head()->EmplaceBack(ObjectMsgPtr<SelfLoopContainer>(self_loop_head0));
    ASSERT_EQ(self_loop_head0->head().size(), 1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    self_loop_head0->mut_head()->EmplaceBack(ObjectMsgPtr<SelfLoopContainer>(self_loop_head1));
    ASSERT_EQ(self_loop_head1->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head0->head().size(), 2);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(ObjectMsgSelfLoopList, EmplaceFront) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
    self_loop_head0->mut_head()->EmplaceFront(ObjectMsgPtr<SelfLoopContainer>(self_loop_head0));
    ASSERT_EQ(self_loop_head0->head().size(), 1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    self_loop_head0->mut_head()->EmplaceFront(ObjectMsgPtr<SelfLoopContainer>(self_loop_head1));
    ASSERT_EQ(self_loop_head1->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head0->head().size(), 2);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(ObjectMsgSelfLoopList, Erase) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->Erase(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->Erase(self_loop_head1.Mutable());
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(ObjectMsgSelfLoopList, PopBack) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->PopBack();
    self_loop_head0->mut_head()->PopBack();
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(ObjectMsgSelfLoopList, PopFront) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->PopFront();
    self_loop_head0->mut_head()->PopFront();
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(ObjectMsgSelfLoopList, MoveTo) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->MoveTo(self_loop_head1->mut_head());
    ASSERT_EQ(self_loop_head0->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(ObjectMsgSelfLoopList, Clear) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted0);
    auto self_loop_head1 = ObjectMsgPtr<SelfLoopContainer>::New(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->Clear();
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
