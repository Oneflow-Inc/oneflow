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
#include "oneflow/core/common/util.h"
#include "oneflow/core/intrusive/intrusive.h"

namespace oneflow {

namespace test {

namespace {

// clang-format off
REFLECTIVE_CLASS_BEGIN(SelfLoopContainer);
 public:
  void __Init__() { clear_deleted(); }
  // Getters
  bool has_deleted() const { return deleted_ != nullptr; }
  bool deleted() const { return *deleted_; } 
  bool is_hook_empty() const { return hook_.empty(); }
  // Setters
  bool* mut_deleted() { return deleted_; }
  void set_deleted(bool* val) { deleted_ = val; }
  void clear_deleted() { deleted_ = nullptr; }

  // methods
  void __Init__(bool* deleted) {
    __Init__();
    set_deleted(deleted);
  }
  void __Delete__() { *mut_deleted() = true; }

  size_t ref_cnt() const { return intrusive_ref_.ref_cnt(); }

 private:
  friend class intrusive::Ref;
  intrusive::Ref* mut_intrusive_ref() { return &intrusive_ref_; }

  SelfLoopContainer() : intrusive_ref_(), deleted_(), hook_(), head_() {}
  REFLECTIVE_CLASS_DEFINE_FIELD(intrusive::Ref, intrusive_ref_);
  // fields
  REFLECTIVE_CLASS_DEFINE_FIELD(bool*, deleted_);
  // list hooks
  REFLECTIVE_CLASS_DEFINE_FIELD(intrusive::ListHook, hook_);

 public:
  // Do not insert other REFLECTIVE_CLASS_DEFINE_FIELD between `using SelfLoopContainerList = ...;` and `REFLECTIVE_CLASS_DEFINE_FIELD(SelfLoopContainerList, ...);` 
  using SelfLoopContainerList =
      intrusive::HeadFreeList<REFLECTIVE_FIELD(SelfLoopContainer, hook_), REFLECTIVE_FIELD_COUNTER>;
  const SelfLoopContainerList& head() const { return head_; }
  SelfLoopContainerList* mut_head() { return &head_; }

 private:
  REFLECTIVE_CLASS_DEFINE_FIELD(SelfLoopContainerList, head_);
REFLECTIVE_CLASS_END(SelfLoopContainer);
// clang-format on

TEST(HeadFreeList, __Init__) {
  bool deleted = false;
  auto self_loop_head = intrusive::make_shared<SelfLoopContainer>(&deleted);
  ASSERT_EQ(self_loop_head->mut_head()->container_, self_loop_head.Mutable());
}

TEST(HeadFreeList, PushBack) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    ASSERT_EQ(self_loop_head0->head().size(), 1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    ASSERT_EQ(self_loop_head1->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head0->head().size(), 2);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(HeadFreeList, PushFront) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
    self_loop_head0->mut_head()->PushFront(self_loop_head0.Mutable());
    ASSERT_EQ(self_loop_head0->head().size(), 1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    self_loop_head0->mut_head()->PushFront(self_loop_head1.Mutable());
    ASSERT_EQ(self_loop_head1->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head0->head().size(), 2);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(HeadFreeList, EmplaceBack) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
    self_loop_head0->mut_head()->EmplaceBack(
        intrusive::shared_ptr<SelfLoopContainer>(self_loop_head0));
    ASSERT_EQ(self_loop_head0->head().size(), 1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    self_loop_head0->mut_head()->EmplaceBack(
        intrusive::shared_ptr<SelfLoopContainer>(self_loop_head1));
    ASSERT_EQ(self_loop_head1->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head0->head().size(), 2);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(HeadFreeList, EmplaceFront) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
    self_loop_head0->mut_head()->EmplaceFront(
        intrusive::shared_ptr<SelfLoopContainer>(self_loop_head0));
    ASSERT_EQ(self_loop_head0->head().size(), 1);
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    self_loop_head0->mut_head()->EmplaceFront(
        intrusive::shared_ptr<SelfLoopContainer>(self_loop_head1));
    ASSERT_EQ(self_loop_head1->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head0->head().size(), 2);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(HeadFreeList, Erase) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->Erase(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->Erase(self_loop_head1.Mutable());
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(HeadFreeList, PopBack) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->PopBack();
    self_loop_head0->mut_head()->PopBack();
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(HeadFreeList, PopFront) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->PopFront();
    self_loop_head0->mut_head()->PopFront();
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(HeadFreeList, MoveTo) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->MoveTo(self_loop_head1->mut_head());
    ASSERT_EQ(self_loop_head0->ref_cnt(), 2);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

TEST(HeadFreeList, Clear) {
  bool deleted0 = false;
  bool deleted1 = false;
  {
    auto self_loop_head0 = intrusive::make_shared<SelfLoopContainer>(&deleted0);
    auto self_loop_head1 = intrusive::make_shared<SelfLoopContainer>(&deleted1);
    self_loop_head0->mut_head()->PushBack(self_loop_head0.Mutable());
    self_loop_head0->mut_head()->PushBack(self_loop_head1.Mutable());
    self_loop_head0->mut_head()->Clear();
    ASSERT_EQ(self_loop_head0->ref_cnt(), 1);
    ASSERT_EQ(self_loop_head1->ref_cnt(), 1);
  }
  ASSERT_TRUE(deleted0);
  ASSERT_TRUE(deleted1);
}

}  // namespace

}  // namespace test

}  // namespace oneflow
