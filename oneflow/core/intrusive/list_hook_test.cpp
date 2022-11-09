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
#include "oneflow/core/intrusive/list_hook.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace intrusive {

namespace test {

struct ListItemBar final {
  ListItemBar() : value() { bar_list.__Init__(); }
  int value;
  ListHook bar_list;
};

class TestListHook final : public ListHook {
 public:
  TestListHook() { this->__Init__(); }
};

template<typename ItemField>
class TestListHead : public intrusive::ListHead<ItemField> {
 public:
  TestListHead() { this->__Init__(); }
};

using BarListHead = TestListHead<INTRUSIVE_FIELD(ListItemBar, bar_list)>;

TEST(TestListHook, init) {
  TestListHook list_iterator;
  ASSERT_EQ(&list_iterator, list_iterator.prev());
  ASSERT_EQ(&list_iterator, list_iterator.next());
}

TEST(TestListHook, append_to) {
  TestListHook list_iter0;
  TestListHook list_iter1;
  list_iter1.AppendTo(&list_iter0);
  ASSERT_EQ(&list_iter0, list_iter1.prev());
  ASSERT_EQ(&list_iter1, list_iter0.next());
}

TEST(TestListHook, clear) {
  TestListHook list_head0;
  TestListHook list_head1;
  list_head1.AppendTo(&list_head0);
  list_head1.__Init__();
  ASSERT_EQ(&list_head1, list_head1.prev());
  ASSERT_EQ(&list_head1, list_head1.next());
}

TEST(ListHead, empty) {
  BarListHead list_head;
  ASSERT_TRUE(list_head.empty());
}

TEST(ListHead, push_front) {
  BarListHead list_head;
  ListHook& head = list_head.container_;
  ListItemBar item0;
  list_head.PushFront(&item0);
  ASSERT_EQ(head.next(), &item0.bar_list);
  ASSERT_EQ(head.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &head);
  ASSERT_EQ(item0.bar_list.prev(), &head);
  ListItemBar item1;
  list_head.PushFront(&item1);
  ASSERT_EQ(head.next(), &item1.bar_list);
  ASSERT_EQ(item1.bar_list.prev(), &head);
  ASSERT_EQ(item1.bar_list.next(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.prev(), &item1.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &head);
  ASSERT_EQ(head.prev(), &item0.bar_list);
}

TEST(ListHead, end) {
  BarListHead list_head;
  ListItemBar* end_item = list_head.End();
  ListItemBar item0;
  list_head.PushFront(&item0);
  ASSERT_EQ(end_item, list_head.End());
}

TEST(ListHead, begin) {
  BarListHead list_head;
  ASSERT_EQ(list_head.Begin(), list_head.End());
  ListItemBar item0;
  list_head.PushFront(&item0);
  ASSERT_EQ(list_head.Begin(), &item0);
  ListItemBar item1;
  list_head.PushFront(&item1);
  ASSERT_EQ(list_head.Begin(), &item1);
}

TEST(ListHead, last) {
  BarListHead list_head;
  ASSERT_EQ(list_head.Begin(), list_head.End());
  ListItemBar item0;
  list_head.PushFront(&item0);
  ASSERT_EQ(list_head.Last(), &item0);
  ListItemBar item1;
  list_head.PushFront(&item1);
  ASSERT_EQ(list_head.Last(), &item0);
}

TEST(ListHead, push_back) {
  BarListHead list_head;
  ASSERT_EQ(list_head.Begin(), list_head.End());
  ListItemBar item0;
  list_head.PushBack(&item0);
  ASSERT_EQ(list_head.Last(), &item0);
  ListItemBar item1;
  list_head.PushBack(&item1);
  ASSERT_EQ(list_head.Last(), &item1);
}

TEST(ListHead, erase) {
  BarListHead list_head;
  ASSERT_EQ(list_head.Begin(), list_head.End());
  ListItemBar item0;
  list_head.PushBack(&item0);
  ASSERT_EQ(list_head.Last(), &item0);
  ListItemBar item1;
  list_head.PushBack(&item1);
  ASSERT_EQ(list_head.Last(), &item1);
  list_head.Erase(&item0);
  ASSERT_EQ(list_head.Last(), &item1);
  ASSERT_EQ(list_head.Begin(), &item1);
  ASSERT_EQ(item0.bar_list.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &item0.bar_list);
}

TEST(ListHead, pop_front) {
  BarListHead list_head;
  ASSERT_EQ(list_head.Begin(), list_head.End());
  ListItemBar item0;
  list_head.PushBack(&item0);
  ASSERT_EQ(list_head.Last(), &item0);
  ListItemBar item1;
  list_head.PushBack(&item1);
  ASSERT_EQ(list_head.Last(), &item1);
  list_head.PopFront();
  ASSERT_EQ(list_head.Last(), &item1);
  ASSERT_EQ(list_head.Begin(), &item1);
  ASSERT_EQ(item0.bar_list.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &item0.bar_list);
}

TEST(ListHead, pop_back) {
  BarListHead list_head;
  ASSERT_EQ(list_head.Begin(), list_head.End());
  ListItemBar item0;
  list_head.PushBack(&item0);
  ASSERT_EQ(list_head.Last(), &item0);
  ListItemBar item1;
  list_head.PushBack(&item1);
  ASSERT_EQ(list_head.Last(), &item1);
  list_head.PopBack();
  ASSERT_EQ(list_head.Last(), &item0);
  ASSERT_EQ(list_head.Begin(), &item0);
  ASSERT_EQ(item1.bar_list.prev(), &item1.bar_list);
  ASSERT_EQ(item1.bar_list.next(), &item1.bar_list);
}

TEST(ListHead, Next) {
  BarListHead list_head;
  ListItemBar item0;
  list_head.PushBack(&item0);
  ListItemBar item1;
  list_head.PushBack(&item1);

  ListItemBar* item = list_head.Begin();
  ASSERT_EQ(item, &item0);
  item = list_head.Next(item);
  ASSERT_EQ(item, &item1);
  item = list_head.Next(item);
  ASSERT_EQ(item, list_head.End());
  item = list_head.Next(item);
  ASSERT_EQ(item, &item0);
}

TEST(ListHead, prev_item) {
  BarListHead list_head;
  ListItemBar item0;
  list_head.PushBack(&item0);
  ListItemBar item1;
  list_head.PushBack(&item1);

  ListItemBar* item = list_head.Begin();
  ASSERT_EQ(item, &item0);
  item = list_head.Prev(item);
  ASSERT_EQ(item, list_head.End());
  item = list_head.Prev(item);
  ASSERT_EQ(item, &item1);
  item = list_head.Prev(item);
  ASSERT_EQ(item, &item0);
}

}  // namespace test

}  // namespace intrusive

}  // namespace oneflow
