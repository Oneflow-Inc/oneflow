#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg.h"

namespace oneflow {

namespace test {

// clang-format off
BEGIN_OBJECT_MSG(TestListItem)
  OBJECT_MSG_DEFINE_LIST_LINK(foo_list);
  OBJECT_MSG_DEFINE_RAW_PTR(int*, cnt);

 public:
  void __Delete__() {
    if (has_cnt()) { --*mutable_cnt(); }
  }
END_OBJECT_MSG(TestListItem)
// clang-format on

TEST(ObjectMsgList, empty) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(foo_list.size(), 0);
}

TEST(ObjectMsgList, empty_Begin) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  obj_ptr = foo_list.Begin();
  ASSERT_TRUE(!obj_ptr);
  OBJECT_MSG_PTR(TestListItem) next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(!obj_ptr);
}

TEST(ObjectMsgList, empty_Next) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
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
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushFront(item0.Mutable());
  foo_list.PushFront(item1.Mutable());
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item1);
  ASSERT_TRUE(next == item0);
}

TEST(ObjectMsgList, destructor) {
  int elem_cnt = 2;
  {
    OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
    auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
    item0->set_cnt(&elem_cnt);
    auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
    item1->set_cnt(&elem_cnt);
    foo_list.PushFront(item0.Mutable());
    foo_list.PushFront(item1.Mutable());
  }
  ASSERT_EQ(elem_cnt, 0);
  elem_cnt = 2;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  {
    OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
    item0->set_cnt(&elem_cnt);
    auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
    item1->set_cnt(&elem_cnt);
    foo_list.PushFront(item0.Mutable());
    foo_list.PushFront(item1.Mutable());
  }
  ASSERT_EQ(elem_cnt, 1);
}

TEST(ObjectMsgList, PushBack) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(next == item1);
}

TEST(ObjectMsgList, Erase) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.Erase(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 1);
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(!next);
}

TEST(ObjectMsgList, PopBack) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.PopBack();
  ASSERT_EQ(item1->ref_cnt(), 1);
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(!next);
}

TEST(ObjectMsgList, PopFront) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  foo_list.PopFront();
  ASSERT_EQ(item0->ref_cnt(), 1);
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  obj_ptr = foo_list.Begin();
  next = foo_list.Next(obj_ptr.Mutable());
  ASSERT_TRUE(!next);
}

TEST(ObjectMsgList, Clear) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.Clear();
  ASSERT_TRUE(foo_list.empty());
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(item1->ref_cnt(), 1);
}

TEST(ObjectMsgList, FOR_EACH_UNSAFE) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  int i = 0;
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(&foo_list, item) {
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
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
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

// clang-format off
BEGIN_OBJECT_MSG(TestObjectMsgListHead);
  OBJECT_MSG_DEFINE_LIST_HEAD(TestListItem, foo_list, foo_list);
END_OBJECT_MSG(TestObjectMsgListHead);
// clang-format on

TEST(ObjectMsg, OBJECT_MSG_DEFINE_LIST_HEAD) {
  auto foo_list_head = OBJECT_MSG_PTR(TestObjectMsgListHead)::New();
  auto& foo_list = *foo_list_head->mutable_foo_list();
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
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
BEGIN_OBJECT_MSG(TestObjectMsgListHeadWrapper);
  OBJECT_MSG_DEFINE_OPTIONAL(TestObjectMsgListHead, head);
END_OBJECT_MSG(TestObjectMsgListHeadWrapper);
// clang-format on

TEST(ObjectMsg, nested_list_delete) {
  auto foo_list_head = OBJECT_MSG_PTR(TestObjectMsgListHeadWrapper)::New();
  auto& foo_list = *foo_list_head->mutable_head()->mutable_foo_list();
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  int i = 0;
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(&foo_list, item) {
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
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(item0.Mutable());
  foo_list.PushBack(item1.Mutable());
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

}  // namespace test

}  // namespace oneflow
