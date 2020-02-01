#include "oneflow/core/common/object_msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgSkipListFoo);
  OBJECT_MSG_DEFINE_MAP_KEY(int, foo_map);
  OBJECT_MSG_DEFINE_RAW_PTR(int*, is_deleted);
  void __Delete__() {
    if (has_is_deleted()) { ++*mutable_is_deleted(); }
  }
END_OBJECT_MSG(ObjectMsgSkipListFoo);
// clang-format on

// clang-format off
BEGIN_OBJECT_MSG(ObjectMsgSkipListFooContainer);
  OBJECT_MSG_DEFINE_MAP_HEAD(ObjectMsgSkipListFoo, foo_map);
END_OBJECT_MSG(ObjectMsgSkipListFooContainer);
// clang-format on

TEST(ObjectMsgSkipList, empty) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map) foo_map;
  ASSERT_TRUE(foo_map.empty());
  ASSERT_EQ(foo_map.size(), 0);
}

TEST(ObjectMsgSkipList, insert_naive) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map) foo_map;
  auto elem0 = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(&elem0);
  ASSERT_EQ(foo_map.size(), 1);
  {
    auto searched = foo_map.Find(int(0));
    ASSERT_TRUE(searched == elem0);
  }
  {
    auto searched = foo_map.Find(int(-1));
    ASSERT_TRUE(foo_map.EqualsEnd(searched));
  }
}

TEST(ObjectMsgSkipList, erase_by_key) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map) foo_map;
  auto elem0 = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(&elem0);
  ASSERT_EQ(foo_map.size(), 1);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(int(0));
  ASSERT_EQ(foo_map.size(), 0);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
}

TEST(ObjectMsgSkipList, erase_by_elem) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map) foo_map;
  auto elem0 = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(&elem0);
  ASSERT_EQ(foo_map.size(), 1);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(&elem0);
  ASSERT_EQ(foo_map.size(), 0);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
}

TEST(ObjectMsgSkipList, insert_many) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map) foo_map;
  OBJECT_MSG_PTR(ObjectMsgSkipListFoo) exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(&exists[i]);
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(&elem0);
  ASSERT_EQ(foo_map.size(), 101);
  {
    auto searched = foo_map.Find(int(0));
    ASSERT_TRUE(searched == elem0);
  }
  {
    auto searched = foo_map.Find(int(-1001));
    ASSERT_TRUE(foo_map.EqualsEnd(searched));
  }
  foo_map.Clear();
  ASSERT_TRUE(foo_map.empty());
}

TEST(ObjectMsgSkipList, erase_many_by_key) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map) foo_map;
  OBJECT_MSG_PTR(ObjectMsgSkipListFoo) exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(&exists[i]);
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(&elem0);
  ASSERT_EQ(foo_map.size(), 101);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(int(0));
  ASSERT_EQ(foo_map.size(), 100);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Clear();
  ASSERT_TRUE(foo_map.empty());
}

TEST(ObjectMsgSkipList, erase_many_by_elem) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map) foo_map;
  OBJECT_MSG_PTR(ObjectMsgSkipListFoo) exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(&exists[i]);
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(&elem0);
  ASSERT_EQ(foo_map.size(), 101);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(&elem0);
  ASSERT_EQ(foo_map.size(), 100);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Clear();
  ASSERT_TRUE(foo_map.empty());
}

TEST(ObjectMsgSkipList, MAP_HEAD) {
  int elem_cnt = 0;
  {
    auto foo_map_container = OBJECT_MSG_PTR(ObjectMsgSkipListFooContainer)::New();
    auto& foo_map = *foo_map_container->mutable_foo_map();
    OBJECT_MSG_PTR(ObjectMsgSkipListFoo) exists[100];
    for (int i = 0; i < 100; ++i) {
      exists[i] = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
      int key = i - 50;
      if (key >= 0) { ++key; }
      exists[i]->set_foo_map_key(key);
      exists[i]->set_is_deleted(&elem_cnt);
      foo_map.Insert(&exists[i]);
      ASSERT_TRUE(foo_map.Find(key) == exists[i]);
      ASSERT_EQ(exists[i]->ref_cnt(), 2);
    }
    auto elem0 = OBJECT_MSG_PTR(ObjectMsgSkipListFoo)::New();
    elem0->set_foo_map_key(0);
    elem0->set_is_deleted(&elem_cnt);
    foo_map.Insert(&elem0);
    ASSERT_EQ(foo_map.size(), 101);
    ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
    ASSERT_EQ(elem0->ref_cnt(), 2);
    foo_map.Erase(elem0->foo_map_key());
    ASSERT_EQ(elem0->ref_cnt(), 1);
    ASSERT_EQ(foo_map.size(), 100);
    ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
    foo_map.Clear();
    ASSERT_TRUE(foo_map.empty());
  }
  ASSERT_EQ(elem_cnt, 101);
}
}  // namespace test

}  // namespace oneflow
