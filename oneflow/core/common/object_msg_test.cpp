#include "oneflow/core/common/util.h"
#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/preprocessor.h"

namespace oneflow {

namespace test {

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgFoo)
  OBJECT_MSG_DEFINE_OPTIONAL(int8_t, x);
  OBJECT_MSG_DEFINE_OPTIONAL(int32_t, foo);
  OBJECT_MSG_DEFINE_OPTIONAL(int16_t, bar);
  OBJECT_MSG_DEFINE_OPTIONAL(int64_t, foobar);
  OBJECT_MSG_DEFINE_RAW_PTR(std::string*, is_deleted);

 public:
  void __Delete__();

END_OBJECT_MSG(ObjectMsgFoo)
// clang-format on

void OBJECT_MSG_TYPE(ObjectMsgFoo)::__Delete__() {
  if (mutable_is_deleted()) { *mutable_is_deleted() = "deleted"; }
}

TEST(OBJECT_MSG, naive) {
  auto foo = OBJECT_MSG_PTR(ObjectMsgFoo)::New();
  foo->set_bar(9527);
  ASSERT_TRUE(foo->bar() == 9527);
}

TEST(OBJECT_MSG, __delete__) {
  std::string is_deleted;
  {
    auto foo = OBJECT_MSG_PTR(ObjectMsgFoo)::New();
    foo->set_bar(9527);
    foo->set_is_deleted(&is_deleted);
    ASSERT_EQ(foo->bar(), 9527);
  }
  ASSERT_TRUE(is_deleted == "deleted");
}

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgBar)
  OBJECT_MSG_DEFINE_OPTIONAL(ObjectMsgFoo, foo);
  OBJECT_MSG_DEFINE_RAW_PTR(std::string*, is_deleted);

 public:
  void __Delete__(){
    if (mutable_is_deleted()) { *mutable_is_deleted() = "bar_deleted"; }
  }
END_OBJECT_MSG(ObjectMsgBar)
// clang-format on

TEST(OBJECT_MSG, nested_objects) {
  auto bar = OBJECT_MSG_PTR(ObjectMsgBar)::New();
  bar->mutable_foo()->set_bar(9527);
  ASSERT_TRUE(bar->foo().bar() == 9527);
}

TEST(OBJECT_MSG, nested_delete) {
  std::string bar_is_deleted;
  std::string is_deleted;
  {
    auto bar = OBJECT_MSG_PTR(ObjectMsgBar)::New();
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
BEGIN_OBJECT_MSG(TestScalarOneof)
  OBJECT_MSG_DEFINE_ONEOF(type,
      OBJECT_MSG_ONEOF_FIELD(int32_t, x)
      OBJECT_MSG_ONEOF_FIELD(int64_t, foo));
END_OBJECT_MSG(TestScalarOneof)
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(TestPtrOneof)
  OBJECT_MSG_DEFINE_ONEOF(type,
      OBJECT_MSG_ONEOF_FIELD(ObjectMsgFoo, foo)
      OBJECT_MSG_ONEOF_FIELD(int32_t, int_field));
END_OBJECT_MSG(TestPtrOneof)
// clang-format on

TEST(OBJECT_MSG, oneof_get) {
  auto test_oneof = OBJECT_MSG_PTR(TestPtrOneof)::New();
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
  auto test_oneof = OBJECT_MSG_PTR(TestPtrOneof)::New();
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

TEST(OBJECT_MSG, oneof_clear) {
  auto test_oneof = OBJECT_MSG_PTR(TestPtrOneof)::New();
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
  auto test_oneof = OBJECT_MSG_PTR(TestPtrOneof)::New();
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
BEGIN_FLAT_MSG(FlatMsgDemo)
  FLAT_MSG_DEFINE_ONEOF(type,
      FLAT_MSG_ONEOF_FIELD(int32_t, int32_field)
      FLAT_MSG_ONEOF_FIELD(float, float_field));
END_FLAT_MSG(FlatMsgDemo)
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgContainerDemo)
  OBJECT_MSG_DEFINE_FLAT_MSG(FlatMsgDemo, flat_field);
END_OBJECT_MSG(ObjectMsgContainerDemo)
// clang-format on

TEST(OBJECT_MSG, flat_msg_field) {
  auto obj = OBJECT_MSG_PTR(ObjectMsgContainerDemo)::New();
  ASSERT_TRUE(obj->has_flat_field());
  ASSERT_TRUE(!obj->flat_field().has_int32_field());
  obj->mutable_flat_field()->set_int32_field(33);
  ASSERT_TRUE(obj->flat_field().has_int32_field());
  ASSERT_EQ(obj->flat_field().int32_field(), 33);
}

// clang-format off
BEGIN_OBJECT_MSG(TestListItem)
  OBJECT_MSG_DEFINE_LIST_ITEM(foo_list);
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

TEST(ObjectMsgList, empty_GetBegin) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  foo_list.GetBegin(&obj_ptr);
  ASSERT_TRUE(foo_list.EqualsEnd(obj_ptr));
  OBJECT_MSG_PTR(TestListItem) next;
  foo_list.GetBegin(&obj_ptr, &next);
  ASSERT_TRUE(foo_list.EqualsEnd(obj_ptr));
}

TEST(ObjectMsgList, empty_MoveToNext) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  foo_list.GetBegin(&obj_ptr, &next);
  ASSERT_TRUE(foo_list.EqualsEnd(obj_ptr));
  ASSERT_TRUE(foo_list.EqualsEnd(next));
  foo_list.MoveToNext(&obj_ptr);
  ASSERT_TRUE(foo_list.EqualsEnd(obj_ptr));
  foo_list.MoveToNext(&obj_ptr, &next);
  ASSERT_TRUE(foo_list.EqualsEnd(obj_ptr));
  ASSERT_TRUE(foo_list.EqualsEnd(next));
}

TEST(ObjectMsgList, PushFront) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushFront(&item0);
  foo_list.PushFront(&item1);
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  foo_list.GetBegin(&obj_ptr, &next);
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
    foo_list.PushFront(&item0);
    foo_list.PushFront(&item1);
  }
  ASSERT_EQ(elem_cnt, 0);
  elem_cnt = 2;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  {
    OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
    item0->set_cnt(&elem_cnt);
    auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
    item1->set_cnt(&elem_cnt);
    foo_list.PushFront(&item0);
    foo_list.PushFront(&item1);
  }
  ASSERT_EQ(elem_cnt, 1);
}

TEST(ObjectMsgList, PushBack) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  foo_list.GetBegin(&obj_ptr, &next);
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(next == item1);
}

TEST(ObjectMsgList, Erase) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.Erase(&item1);
  ASSERT_EQ(item1->ref_cnt(), 1);
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  foo_list.GetBegin(&obj_ptr, &next);
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(foo_list.EqualsEnd(next));
}

TEST(ObjectMsgList, PopBack) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
  ASSERT_EQ(item1->ref_cnt(), 2);
  foo_list.PopBack();
  ASSERT_EQ(item1->ref_cnt(), 1);
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  foo_list.GetBegin(&obj_ptr, &next);
  ASSERT_TRUE(obj_ptr == item0);
  ASSERT_TRUE(foo_list.EqualsEnd(next));
}

TEST(ObjectMsgList, PopFront) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
  ASSERT_EQ(item0->ref_cnt(), 2);
  foo_list.PopFront();
  ASSERT_EQ(item0->ref_cnt(), 1);
  OBJECT_MSG_PTR(TestListItem) obj_ptr;
  OBJECT_MSG_PTR(TestListItem) next;
  foo_list.GetBegin(&obj_ptr, &next);
  ASSERT_TRUE(obj_ptr == item1);
  ASSERT_TRUE(foo_list.EqualsEnd(next));
}

TEST(ObjectMsgList, Clear) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
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
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
  int i = 0;
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(&foo_list, item) {
    if (i == 0) {
      ASSERT_TRUE(item == item0);
    } else if (i == 1) {
      ASSERT_TRUE(item == item1);
    }
    ++i;
  }
  ASSERT_EQ(i, 2);
}

TEST(ObjectMsgList, FOR_EACH) {
  OBJECT_MSG_LIST(TestListItem, foo_list) foo_list;
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
  int i = 0;
  OBJECT_MSG_LIST_FOR_EACH(&foo_list, item) {
    if (i == 0) {
      ASSERT_TRUE(item == item0);
      foo_list.Erase(&item);
    } else if (i == 1) {
      ASSERT_TRUE(item == item1);
      foo_list.Erase(&item);
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
  OBJECT_MSG_DEFINE_LIST_HEAD(TestListItem, foo_list);
END_OBJECT_MSG(TestObjectMsgListHead);
// clang-format on

TEST(ObjectMsg, OBJECT_MSG_DEFINE_LIST_HEAD) {
  auto foo_list_head = OBJECT_MSG_PTR(TestObjectMsgListHead)::New();
  auto& foo_list = *foo_list_head->mutable_foo_list();
  auto item0 = OBJECT_MSG_PTR(TestListItem)::New();
  auto item1 = OBJECT_MSG_PTR(TestListItem)::New();
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  int i = 0;
  OBJECT_MSG_LIST_FOR_EACH(&foo_list, item) {
    if (i == 0) {
      ASSERT_TRUE(item == item0);
      foo_list.Erase(&item);
    } else if (i == 1) {
      ASSERT_TRUE(item == item1);
      foo_list.Erase(&item);
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
  foo_list.PushBack(&item0);
  foo_list.PushBack(&item1);
  ASSERT_EQ(item0->ref_cnt(), 2);
  ASSERT_EQ(item1->ref_cnt(), 2);
  int i = 0;
  OBJECT_MSG_LIST_FOR_EACH_UNSAFE(&foo_list, item) {
    if (i == 0) {
      ASSERT_TRUE(item == item0);
    } else if (i == 1) {
      ASSERT_TRUE(item == item1);
    }
    ++i;
  }
  ASSERT_EQ(i, 2);
  foo_list_head->clear_head();
  ASSERT_EQ(item0->ref_cnt(), 1);
  ASSERT_EQ(item1->ref_cnt(), 1);
}

}  // namespace test

}  // namespace oneflow
