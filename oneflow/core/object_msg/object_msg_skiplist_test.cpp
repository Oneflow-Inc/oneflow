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
#include "oneflow/core/object_msg/object_msg.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace test {

namespace {

// clang-format off
OBJECT_MSG_BEGIN(ObjectMsgSkipListFoo);
  OBJECT_MSG_DEFINE_MAP_KEY(int32_t, foo_map_key);
  OBJECT_MSG_DEFINE_PTR(int, is_deleted);
  void __Delete__() {
    if (has_is_deleted()) { ++*mutable_is_deleted(); }
  }
OBJECT_MSG_END(ObjectMsgSkipListFoo);
// clang-format on

// clang-format off
OBJECT_MSG_BEGIN(ObjectMsgSkipListFooContainer);
  OBJECT_MSG_DEFINE_MAP_HEAD(ObjectMsgSkipListFoo, foo_map_key, foo_map);
OBJECT_MSG_END(ObjectMsgSkipListFooContainer);
// clang-format on

TEST(ObjectMsgSkipList, empty) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map_key) foo_map;
  ASSERT_TRUE(foo_map.empty());
  ASSERT_EQ(foo_map.size(), 0);
}

TEST(ObjectMsgSkipList, insert_naive) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map_key) foo_map;
  auto elem0 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(elem0.Mutable());
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

TEST(ObjectMsgSkipList, insert_twice) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map_key) foo_map;
  auto elem0 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
  elem0->set_foo_map_key(0);
  auto elem1 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
  elem1->set_foo_map_key(0);
  ASSERT_TRUE(foo_map.Insert(elem0.Mutable()).second);
  ASSERT_TRUE(!foo_map.Insert(elem1.Mutable()).second);
}

TEST(ObjectMsgSkipList, erase_by_key) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map_key) foo_map;
  auto elem0 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 1);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(int(0));
  ASSERT_EQ(foo_map.size(), 0);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
}

TEST(ObjectMsgSkipList, erase_by_elem) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map_key) foo_map;
  auto elem0 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 1);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 0);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
}

TEST(ObjectMsgSkipList, insert_many) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map_key) foo_map;
  ObjectMsgPtr<ObjectMsgSkipListFoo> exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(exists[i].Mutable());
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(elem0.Mutable());
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
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map_key) foo_map;
  ObjectMsgPtr<ObjectMsgSkipListFoo> exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(exists[i].Mutable());
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 101);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(int(0));
  ASSERT_EQ(foo_map.size(), 100);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Clear();
  ASSERT_TRUE(foo_map.empty());
}

TEST(ObjectMsgSkipList, erase_many_by_elem) {
  OBJECT_MSG_MAP(ObjectMsgSkipListFoo, foo_map_key) foo_map;
  ObjectMsgPtr<ObjectMsgSkipListFoo> exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(exists[i].Mutable());
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
  elem0->set_foo_map_key(0);
  foo_map.Insert(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 101);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 100);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Clear();
  ASSERT_TRUE(foo_map.empty());
}

TEST(ObjectMsgSkipList, MAP_HEAD) {
  int elem_cnt = 0;
  {
    auto foo_map_container = ObjectMsgPtr<ObjectMsgSkipListFooContainer>::New();
    auto& foo_map = *foo_map_container->mutable_foo_map();
    ObjectMsgPtr<ObjectMsgSkipListFoo> exists[100];
    for (int i = 0; i < 100; ++i) {
      exists[i] = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
      int key = i - 50;
      if (key >= 0) { ++key; }
      exists[i]->set_foo_map_key(key);
      exists[i]->set_is_deleted(&elem_cnt);
      foo_map.Insert(exists[i].Mutable());
      ASSERT_TRUE(foo_map.Find(key) == exists[i]);
      ASSERT_EQ(exists[i]->ref_cnt(), 2);
    }
    auto elem0 = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
    elem0->set_foo_map_key(0);
    elem0->set_is_deleted(&elem_cnt);
    foo_map.Insert(elem0.Mutable());
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

TEST(ObjectMsgSkipList, FOR_EACH) {
  int elem_cnt = 0;
  {
    auto foo_map_container = ObjectMsgPtr<ObjectMsgSkipListFooContainer>::New();
    auto& foo_map = *foo_map_container->mutable_foo_map();
    ObjectMsgPtr<ObjectMsgSkipListFoo> exists[100];
    for (int i = 0; i < 100; ++i) {
      exists[i] = ObjectMsgPtr<ObjectMsgSkipListFoo>::New();
      int key = i - 50;
      exists[i]->set_foo_map_key(key);
      exists[i]->set_is_deleted(&elem_cnt);
      foo_map.Insert(exists[i].Mutable());
      ASSERT_TRUE(foo_map.Find(key) == exists[i]);
      ASSERT_EQ(exists[i]->ref_cnt(), 2);
    }
    int value = -50;
    OBJECT_MSG_SKIPLIST_UNSAFE_FOR_EACH_PTR(&foo_map, foo) {
      ASSERT_EQ(foo->foo_map_key(), value);
      ++value;
    }
  }
  ASSERT_EQ(elem_cnt, 100);
}

// clang-format off
OBJECT_MSG_BEGIN(PtrAsKey);
  OBJECT_MSG_DEFINE_SKIPLIST_KEY(10, int*, ptr_key);
OBJECT_MSG_END(PtrAsKey);
// clang-format on

}  // namespace

}  // namespace test

}  // namespace oneflow
