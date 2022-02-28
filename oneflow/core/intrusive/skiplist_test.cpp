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
#include "gtest/gtest.h"
#include "oneflow/core/intrusive/intrusive.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace intrusive {

namespace test {

namespace {

class SkipListFoo final : public intrusive::Base {
 public:
  void __Init__() { clear_is_deleted(); }
  void __Delete__() {
    if (has_is_deleted()) { ++*mut_is_deleted(); }
  }

  // Getters
  bool has_is_deleted() const { return is_deleted_ != nullptr; }
  int is_deleted() const { return *is_deleted_; }
  int32_t foo_map_key() const { return foo_map_key_.key(); }
  // Setters
  void set_is_deleted(int* val) { is_deleted_ = val; }
  void clear_is_deleted() { is_deleted_ = nullptr; }
  int* mut_is_deleted() { return is_deleted_; }
  void set_foo_map_key(int32_t val) { *foo_map_key_.mut_key() = val; }

  size_t ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  SkipListFoo() : intrusive_ref_(), is_deleted_(), foo_map_key_() {}
  intrusive::Ref intrusive_ref_;
  int* is_deleted_;

 public:
  intrusive::SkipListHook<int32_t> foo_map_key_;
};

class SkipListFooContainer final : public intrusive::Base {
 public:
  // types
  using Key2SkipListFoo = intrusive::SkipList<INTRUSIVE_FIELD(SkipListFoo, foo_map_key_)>;
  // Getters
  const Key2SkipListFoo& foo_map() const { return foo_map_; }
  // Setters
  Key2SkipListFoo* mut_foo_map() { return &foo_map_; }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  SkipListFooContainer() : intrusive_ref_(), foo_map_() {}
  intrusive::Ref intrusive_ref_;
  // maps
  Key2SkipListFoo foo_map_;
};

using Key2SkipListFoo = intrusive::SkipList<INTRUSIVE_FIELD(SkipListFoo, foo_map_key_)>;
TEST(SkipList, empty) {
  Key2SkipListFoo foo_map;
  ASSERT_TRUE(foo_map.empty());
  ASSERT_EQ(foo_map.size(), 0);
}

TEST(SkipList, insert_naive) {
  Key2SkipListFoo foo_map;
  auto elem0 = intrusive::make_shared<SkipListFoo>();
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

TEST(SkipList, insert_twice) {
  Key2SkipListFoo foo_map;
  auto elem0 = intrusive::make_shared<SkipListFoo>();
  elem0->set_foo_map_key(0);
  auto elem1 = intrusive::make_shared<SkipListFoo>();
  elem1->set_foo_map_key(0);
  ASSERT_TRUE(foo_map.Insert(elem0.Mutable()).second);
  ASSERT_TRUE(!foo_map.Insert(elem1.Mutable()).second);
}

TEST(SkipList, erase_by_key) {
  Key2SkipListFoo foo_map;
  auto elem0 = intrusive::make_shared<SkipListFoo>();
  elem0->set_foo_map_key(0);
  foo_map.Insert(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 1);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(int(0));
  ASSERT_EQ(foo_map.size(), 0);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
}

TEST(SkipList, erase_by_elem) {
  Key2SkipListFoo foo_map;
  auto elem0 = intrusive::make_shared<SkipListFoo>();
  elem0->set_foo_map_key(0);
  foo_map.Insert(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 1);
  ASSERT_TRUE(!foo_map.EqualsEnd(foo_map.Find(int(0))));
  foo_map.Erase(elem0.Mutable());
  ASSERT_EQ(foo_map.size(), 0);
  ASSERT_TRUE(foo_map.EqualsEnd(foo_map.Find(int(0))));
}

TEST(SkipList, insert_many) {
  Key2SkipListFoo foo_map;
  intrusive::shared_ptr<SkipListFoo> exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = intrusive::make_shared<SkipListFoo>();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(exists[i].Mutable());
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = intrusive::make_shared<SkipListFoo>();
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

TEST(SkipList, erase_many_by_key) {
  Key2SkipListFoo foo_map;
  intrusive::shared_ptr<SkipListFoo> exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = intrusive::make_shared<SkipListFoo>();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(exists[i].Mutable());
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = intrusive::make_shared<SkipListFoo>();
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

TEST(SkipList, erase_many_by_elem) {
  Key2SkipListFoo foo_map;
  intrusive::shared_ptr<SkipListFoo> exists[100];
  for (int i = 0; i < 100; ++i) {
    exists[i] = intrusive::make_shared<SkipListFoo>();
    int key = i - 50;
    if (key >= 0) { ++key; }
    exists[i]->set_foo_map_key(key);
    foo_map.Insert(exists[i].Mutable());
    ASSERT_TRUE(foo_map.Find(key) == exists[i]);
  }
  auto elem0 = intrusive::make_shared<SkipListFoo>();
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

TEST(SkipList, MAP_HEAD) {
  int elem_cnt = 0;
  {
    auto foo_map_container = intrusive::make_shared<SkipListFooContainer>();
    auto& foo_map = *foo_map_container->mut_foo_map();
    intrusive::shared_ptr<SkipListFoo> exists[100];
    for (int i = 0; i < 100; ++i) {
      exists[i] = intrusive::make_shared<SkipListFoo>();
      int key = i - 50;
      if (key >= 0) { ++key; }
      exists[i]->set_foo_map_key(key);
      exists[i]->set_is_deleted(&elem_cnt);
      foo_map.Insert(exists[i].Mutable());
      ASSERT_TRUE(foo_map.Find(key) == exists[i]);
      ASSERT_EQ(exists[i]->ref_cnt(), 2);
    }
    auto elem0 = intrusive::make_shared<SkipListFoo>();
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

TEST(SkipList, FOR_EACH) {
  int elem_cnt = 0;
  {
    auto foo_map_container = intrusive::make_shared<SkipListFooContainer>();
    auto& foo_map = *foo_map_container->mut_foo_map();
    intrusive::shared_ptr<SkipListFoo> exists[100];
    for (int i = 0; i < 100; ++i) {
      exists[i] = intrusive::make_shared<SkipListFoo>();
      int key = i - 50;
      exists[i]->set_foo_map_key(key);
      exists[i]->set_is_deleted(&elem_cnt);
      foo_map.Insert(exists[i].Mutable());
      ASSERT_TRUE(foo_map.Find(key) == exists[i]);
      ASSERT_EQ(exists[i]->ref_cnt(), 2);
    }
    int value = -50;
    INTRUSIVE_UNSAFE_FOR_EACH_PTR(foo, &foo_map) {
      ASSERT_EQ(foo->foo_map_key(), value);
      ++value;
    }
  }
  ASSERT_EQ(elem_cnt, 100);
}

}  // namespace

}  // namespace test

}  // namespace intrusive

}  // namespace oneflow
