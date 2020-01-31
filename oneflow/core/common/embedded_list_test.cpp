#define private public
#include "oneflow/core/common/embedded_list.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

struct ListItemBar final {
  ListItemBar() { bar_list.__Init__(); }
  int value;
  EmbeddedListItem bar_list;
};

class TestEmbeddedListItem final : public EmbeddedListItem {
 public:
  TestEmbeddedListItem() { this->__Init__(); }
};

template<typename ItemField>
class TestEmbeddedList : public EmbeddedListHead<ItemField> {
 public:
  TestEmbeddedList() { this->__Init__(); }
};

using BarListView = TestEmbeddedList<STRUCT_FIELD(ListItemBar, bar_list)>;

TEST(TestEmbeddedListItem, init) {
  TestEmbeddedListItem list_iterator;
  ASSERT_EQ(&list_iterator, list_iterator.prev());
  ASSERT_EQ(&list_iterator, list_iterator.next());
}

TEST(TestEmbeddedListItem, append_to) {
  TestEmbeddedListItem list_iter0;
  TestEmbeddedListItem list_iter1;
  list_iter1.AppendTo(&list_iter0);
  ASSERT_EQ(&list_iter0, list_iter1.prev());
  ASSERT_EQ(&list_iter1, list_iter0.next());
}

TEST(TestEmbeddedListItem, clear) {
  TestEmbeddedListItem list_head0;
  TestEmbeddedListItem list_head1;
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
  EmbeddedListItem& head = list_view.container_;
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
  ListItemBar* end_item = list_view.end_item();
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(end_item, list_view.end_item());
}

TEST(EmbeddedListView, begin) {
  BarListView list_view;
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(list_view.begin_item(), &item0);
  ListItemBar item1;
  list_view.PushFront(&item1);
  ASSERT_EQ(list_view.begin_item(), &item1);
}

TEST(EmbeddedListView, last) {
  BarListView list_view;
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushFront(&item0);
  ASSERT_EQ(list_view.last_item(), &item0);
  ListItemBar item1;
  list_view.PushFront(&item1);
  ASSERT_EQ(list_view.last_item(), &item0);
}

TEST(EmbeddedListView, push_back) {
  BarListView list_view;
  ASSERT_EQ(list_view.begin_item(), list_view.end_item());
  ListItemBar item0;
  list_view.PushBack(&item0);
  ASSERT_EQ(list_view.last_item(), &item0);
  ListItemBar item1;
  list_view.PushBack(&item1);
  ASSERT_EQ(list_view.last_item(), &item1);
}

TEST(EmbeddedListView, erase) {
  BarListView list_view;
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
  BarListView list_view;
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
  BarListView list_view;
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
  BarListView list_view;
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
  BarListView list_view;
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
