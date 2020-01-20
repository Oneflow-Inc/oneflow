#include "oneflow/core/common/list_head.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

struct ListHeadFoo final {
  int head_field_0;
  int head_field_1;
  ListHead bar_list;
};

struct ListItemBar final {
  int value;
  ListHead bar_list;
};

}  // namespace test

DEFINE_EMBEDDED_LIST_VIEW(test::ListHeadFoo, bar_list, test::ListItemBar, bar_list);
using BarListView = EMBEDDED_LIST_VIEW(test::ListHeadFoo, bar_list);

namespace test {

TEST(ListHead, init) {
  ListHead list_head;
  ASSERT_EQ(&list_head, list_head.prev());
  ASSERT_EQ(&list_head, list_head.next());
}

TEST(ListHead, append_to) {
  ListHead list_head0;
  ListHead list_head1;
  list_head1.AppendTo(&list_head0);
  ASSERT_EQ(&list_head0, list_head1.prev());
  ASSERT_EQ(&list_head1, list_head0.next());
}

TEST(ListHead, clear) {
  ListHead list_head0;
  ListHead list_head1;
  list_head1.AppendTo(&list_head0);
  list_head1.Clear();
  ASSERT_EQ(&list_head1, list_head1.prev());
  ASSERT_EQ(&list_head1, list_head1.next());
}

TEST(EmbeddedListView, empty) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ASSERT_TRUE(list_view.empty());
}

TEST(EmbeddedListView, push_front) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(head.bar_list.next(), &item0.bar_list);
  ASSERT_EQ(head.bar_list.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &head.bar_list);
  ASSERT_EQ(item0.bar_list.prev(), &head.bar_list);
  ListItemBar item1;
  list_view.PushFront(&item1);
  ASSERT_EQ(head.bar_list.next(), &item1.bar_list);
  ASSERT_EQ(item1.bar_list.prev(), &head.bar_list);
  ASSERT_EQ(item1.bar_list.next(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.prev(), &item1.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &head.bar_list);
  ASSERT_EQ(head.bar_list.prev(), &item0.bar_list);
}

TEST(EmbeddedListView, end) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ListItemBar* end_item = list_view.end_item();
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(end_item, list_view.end_item());
}

TEST(EmbeddedListView, begin) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(list_view.begin_item(), &item0);
  ListItemBar item1;
  list_view.PushFront(&item1);
  ASSERT_EQ(list_view.begin_item(), &item1);
}

TEST(EmbeddedListView, last) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(list_view.last_item(), &item0);
  ListItemBar item1;
  list_view.PushFront(&item1);
  ASSERT_EQ(list_view.last_item(), &item0);
}

TEST(EmbeddedListView, push_back) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.last_item(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.last_item(), &item1);
}

TEST(EmbeddedListView, erase) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.last_item(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.last_item(), &item1);
  list_view.Erase(&item0);
  ASSERT_EQ(list_view.last_item(), &item1);
  ASSERT_EQ(list_view.begin_item(), &item1);
  ASSERT_EQ(item0.bar_list.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &item0.bar_list);
}

TEST(EmbeddedListView, pop_front) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.last_item(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.last_item(), &item1);
  list_view.PopFront();
  ASSERT_EQ(list_view.last_item(), &item1);
  ASSERT_EQ(list_view.begin_item(), &item1);
  ASSERT_EQ(item0.bar_list.prev(), &item0.bar_list);
  ASSERT_EQ(item0.bar_list.next(), &item0.bar_list);
}

TEST(EmbeddedListView, pop_back) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.last_item(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.last_item(), &item1);
  list_view.PopBack();
  ASSERT_EQ(list_view.last_item(), &item0);
  ASSERT_EQ(list_view.begin_item(), &item0);
  ASSERT_EQ(item1.bar_list.prev(), &item1.bar_list);
  ASSERT_EQ(item1.bar_list.next(), &item1.bar_list);
}

TEST(EmbeddedListView, next_item) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ListItemBar item0;
  list_view.PushBack(&item0);
  ListItemBar item1;
  list_view.PushBack(&item1);

  ListItemBar* item = list_view.begin_item();
  ASSERT_EQ(item, &item0);
  item = list_view.next_item(item);
  ASSERT_EQ(item, &item1);
  item = list_view.next_item(item);
  ASSERT_EQ(item, list_view.end_item());
  item = list_view.next_item(item);
  ASSERT_EQ(item, &item0);
}

TEST(EmbeddedListView, prev_item) {
  ListHeadFoo head;
  BarListView list_view(&head);
  ListItemBar item0;
  list_view.PushBack(&item0);
  ListItemBar item1;
  list_view.PushBack(&item1);

  ListItemBar* item = list_view.begin_item();
  ASSERT_EQ(item, &item0);
  item = list_view.prev_item(item);
  ASSERT_EQ(item, list_view.end_item());
  item = list_view.prev_item(item);
  ASSERT_EQ(item, &item1);
  item = list_view.prev_item(item);
  ASSERT_EQ(item, &item0);
}

}  // namespace test

}  // namespace oneflow
