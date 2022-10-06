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
#include "oneflow/core/intrusive/skiplist_hook.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

namespace intrusive {

namespace test {

template<typename ElemKeyField>
class TestSkipListHead final : public SkipListHead<ElemKeyField> {  // NOLINT
 public:
  TestSkipListHead() { this->__Init__(); }
  TestSkipListHead(const TestSkipListHead&) = delete;
  TestSkipListHead(TestSkipListHead&&) = delete;
  TestSkipListHead& operator==(const TestSkipListHead&) = delete;
  TestSkipListHead& operator==(TestSkipListHead&&) = delete;
  ~TestSkipListHead() { this->Clear(); }
};

struct FooSkipListElem {
  FooSkipListElem() : value() { key.__Init__(); }

  int value;
  SkipListHook<int> key;
};

using FooSkipList = TestSkipListHead<INTRUSIVE_FIELD(FooSkipListElem, key)>;

TEST(SkipListHook, empty) {
  FooSkipList skiplist;
  ASSERT_TRUE(skiplist.empty_debug());
  ASSERT_EQ(skiplist.size(), 0);
}

TEST(SkipListHook, insert_naive) {
  FooSkipList skiplist;
  FooSkipListElem elem0;
  *elem0.key.mut_key() = 0;
  elem0.value = 1;
  skiplist.Insert(&elem0);
  ASSERT_EQ(skiplist.size(), 1);
  {
    auto* searched = skiplist.Find(int(0));
    ASSERT_EQ(searched, &elem0);
  }
  {
    auto* searched = skiplist.Find(int(-1));
    ASSERT_TRUE(searched == nullptr);
  }
}

TEST(SkipListHook, erase_by_key) {
  FooSkipList skiplist;
  FooSkipListElem elem0;
  *elem0.key.mut_key() = 0;
  elem0.value = 1;
  skiplist.Insert(&elem0);
  ASSERT_EQ(skiplist.size(), 1);
  ASSERT_TRUE(skiplist.Find(int(0)) != nullptr);
  skiplist.Erase(int(0));
  ASSERT_EQ(skiplist.size(), 0);
  ASSERT_TRUE(skiplist.Find(int(0)) == nullptr);
}

TEST(SkipListHook, erase_by_elem) {
  FooSkipList skiplist;
  FooSkipListElem elem0;
  *elem0.key.mut_key() = 0;
  elem0.value = 1;
  skiplist.Insert(&elem0);
  ASSERT_EQ(skiplist.size(), 1);
  ASSERT_TRUE(skiplist.Find(int(0)) != nullptr);
  skiplist.Erase(&elem0);
  ASSERT_EQ(skiplist.size(), 0);
  ASSERT_TRUE(skiplist.Find(int(0)) == nullptr);
}

TEST(SkipListHook, insert_many) {
  FooSkipList skiplist;
  FooSkipListElem exists[100];
  for (int i = 0; i < 100; ++i) {
    int key = i - 50;
    if (key >= 0) { ++key; }
    *exists[i].key.mut_key() = key;
    skiplist.Insert(&exists[i]);
    ASSERT_EQ(skiplist.Find(key), &exists[i]);
  }
  FooSkipListElem elem0;
  *elem0.key.mut_key() = 0;
  elem0.value = 1;
  skiplist.Insert(&elem0);
  ASSERT_EQ(skiplist.size(), 101);
  {
    auto* searched = skiplist.Find(int(0));
    ASSERT_EQ(searched, &elem0);
  }
  {
    auto* searched = skiplist.Find(int(-1001));
    ASSERT_TRUE(searched == nullptr);
  }
  skiplist.Clear();
  ASSERT_TRUE(skiplist.empty_debug());
}

TEST(SkipListHook, erase_many_by_key) {
  FooSkipList skiplist;
  FooSkipListElem exists[100];
  for (int i = 0; i < 100; ++i) {
    int key = i - 50;
    if (key >= 0) { ++key; }
    *exists[i].key.mut_key() = key;
    skiplist.Insert(&exists[i]);
    ASSERT_EQ(skiplist.Find(key), &exists[i]);
  }
  FooSkipListElem elem0;
  *elem0.key.mut_key() = 0;
  elem0.value = 1;
  skiplist.Insert(&elem0);
  ASSERT_EQ(skiplist.size(), 101);
  ASSERT_TRUE(skiplist.Find(int(0)) != nullptr);
  skiplist.Erase(int(0));
  ASSERT_EQ(skiplist.size(), 100);
  ASSERT_TRUE(skiplist.Find(int(0)) == nullptr);
  skiplist.Clear();
  ASSERT_TRUE(skiplist.empty_debug());
}

TEST(SkipListHook, erase_many_by_elem) {
  FooSkipList skiplist;
  FooSkipListElem exists[100];
  for (int i = 0; i < 100; ++i) {
    int key = i - 50;
    if (key >= 0) { ++key; }
    *exists[i].key.mut_key() = key;
    skiplist.Insert(&exists[i]);
    ASSERT_EQ(skiplist.Find(key), &exists[i]);
  }
  FooSkipListElem elem0;
  *elem0.key.mut_key() = 0;
  elem0.value = 1;
  skiplist.Insert(&elem0);
  ASSERT_EQ(skiplist.size(), 101);
  ASSERT_TRUE(skiplist.Find(int(0)) != nullptr);
  skiplist.Erase(&elem0);
  ASSERT_EQ(skiplist.size(), 100);
  ASSERT_TRUE(skiplist.Find(int(0)) == nullptr);
  skiplist.Clear();
  ASSERT_TRUE(skiplist.empty_debug());
}

}  // namespace test

}  // namespace intrusive

}  // namespace oneflow
