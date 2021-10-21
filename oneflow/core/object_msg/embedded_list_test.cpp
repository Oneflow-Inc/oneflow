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
#include "oneflow/core/object_msg/embedded_list.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

struct ListItemBar final {
  ListItemBar() { bar_list.__Init__(); }
  int value;
  EmbeddedListLink bar_list;
};

class TestEmbeddedListLink final : public EmbeddedListLink {
 public:
  TestEmbeddedListLink() { this->__Init__(); }
};

template<typename ItemField>
class TestEmbeddedList : public EmbeddedListHead<ItemField> {
 public:
  TestEmbeddedList() { this->__Init__(); }
};

using BarListView = TestEmbeddedList<STRUCT_FIELD(ListItemBar, bar_list)>;

TEST(TestEmbeddedListLink, init) {
  TestEmbeddedListLink list_iterator;
  ASSERT_EQ(&list_iterator, list_iterator.prev());
  ASSERT_EQ(&list_iterator, list_iterator.next());
}

TEST(TestEmbeddedListLink, append_to) {
  TestEmbeddedListLink list_iter0;
  TestEmbeddedListLink list_iter1;
  list_iter1.AppendTo(&list_iter0);
  ASSERT_EQ(&list_iter0, list_iter1.prev());
  ASSERT_EQ(&list_iter1, list_iter0.next());
}

TEST(TestEmbeddedListLink, clear) {
  TestEmbeddedListLink list_head0;
  TestEmbeddedListLink list_head1;
  list_head1.AppendTo(&list_head0);
  list_head1.__Init__();
  ASSERT_EQ(&list_head1, list_head1.prev());
  ASSERT_EQ(&list_head1, list_head1.next());
}

TEST(EmbeddedListView, empty) {
  BarListView list_view;
  ASSERT_TRUE(list_view.empty());
}

TEST(EmbeddedListView, push_front) {
  BarListView list_view;
  EmbeddedListLink& head = list_view.container_;
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(head.next(), &item0.bar_list);
  ASSERT_EQ(head.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &head);
  ASSERT_EQ(item0.bar_list.prev(), &head);
  ListItemBar item1;
  list_view.PushFront(&item1);
  ASSERT_EQ(head.next(), &item1.bar_list);
  ASSERT_EQ(item1.bar_list.prev(), &head);
  ASSERT_EQ(item1.bar_list.next(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.prev(), &item1.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &head);
  ASSERT_EQ(head.prev(), &item0.bar_list);
}

TEST(EmbeddedListView, end) {
  BarListView list_view;
  ListItemBar* end_item = list_view.End();
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(end_item, list_view.End());
}

TEST(EmbeddedListView, begin) {
  BarListView list_view;
  ASSERT_EQ(list_view.Begin(), list_view.End());
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(list_view.Begin(), &item0);
  ListItemBar item1;
  list_view.PushFront(&item1);
  ASSERT_EQ(list_view.Begin(), &item1);
}

TEST(EmbeddedListView, last) {
  BarListView list_view;
  ASSERT_EQ(list_view.Begin(), list_view.End());
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(list_view.Last(), &item0);
  ListItemBar item1;
  list_view.PushFront(&item1);
  ASSERT_EQ(list_view.Last(), &item0);
}

TEST(EmbeddedListView, push_back) {
  BarListView list_view;
  ASSERT_EQ(list_view.Begin(), list_view.End());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.Last(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.Last(), &item1);
}

TEST(EmbeddedListView, erase) {
  BarListView list_view;
  ASSERT_EQ(list_view.Begin(), list_view.End());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.Last(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.Last(), &item1);
  list_view.Erase(&item0);
  ASSERT_EQ(list_view.Last(), &item1);
  ASSERT_EQ(list_view.Begin(), &item1);
  ASSERT_EQ(item0.bar_list.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &item0.bar_list);
}

TEST(EmbeddedListView, pop_front) {
  BarListView list_view;
  ASSERT_EQ(list_view.Begin(), list_view.End());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.Last(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.Last(), &item1);
  list_view.PopFront();
  ASSERT_EQ(list_view.Last(), &item1);
  ASSERT_EQ(list_view.Begin(), &item1);
  ASSERT_EQ(item0.bar_list.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &item0.bar_list);
}

TEST(EmbeddedListView, pop_back) {
  BarListView list_view;
  ASSERT_EQ(list_view.Begin(), list_view.End());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.Last(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.Last(), &item1);
  list_view.PopBack();
  ASSERT_EQ(list_view.Last(), &item0);
  ASSERT_EQ(list_view.Begin(), &item0);
  ASSERT_EQ(item1.bar_list.prev(), &item1.bar_list);
  ASSERT_EQ(item1.bar_list.next(), &item1.bar_list);
}

TEST(EmbeddedListView, Next) {
  BarListView list_view;
  ListItemBar item0;
  list_view.PushBack(&item0);
  ListItemBar item1;
  list_view.PushBack(&item1);

  ListItemBar* item = list_view.Begin();
  ASSERT_EQ(item, &item0);
  item = list_view.Next(item);
  ASSERT_EQ(item, &item1);
  item = list_view.Next(item);
  ASSERT_EQ(item, list_view.End());
  item = list_view.Next(item);
  ASSERT_EQ(item, &item0);
}

TEST(EmbeddedListView, prev_item) {
  BarListView list_view;
  ListItemBar item0;
  list_view.PushBack(&item0);
  ListItemBar item1;
  list_view.PushBack(&item1);

  ListItemBar* item = list_view.Begin();
  ASSERT_EQ(item, &item0);
  item = list_view.Prev(item);
  ASSERT_EQ(item, list_view.End());
  item = list_view.Prev(item);
  ASSERT_EQ(item, &item1);
  item = list_view.Prev(item);
  ASSERT_EQ(item, &item0);
}

}  // namespace test

}  // namespace oneflow
