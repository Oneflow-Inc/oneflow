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
#ifndef ONEFLOW_CORE_INTRUSIVE_EMBEDDED_SKIPLIST_H_
#define ONEFLOW_CORE_INTRUSIVE_EMBEDDED_SKIPLIST_H_

#include <array>
#include <tuple>
#include <random>
#include <glog/logging.h>
#include "oneflow/core/intrusive/struct_traits.h"
#include "oneflow/core/intrusive/list_hook.h"

namespace oneflow {

namespace intrusive {

template<int max_level>
struct ListHookArray final {
 public:
  ListHookArray() { Clear(); }
  using self_type = ListHookArray<max_level>;
  template<typename Enabled = void>
  static constexpr int LevelZeroHookOffset() {
    return 0;
  }

  bool empty() const { return hooks_[0].nullptr_empty(); }

  void __Init__() { Clear(); }

  void Clear() {
    for (auto& hook : hooks_) { hook.Clear(); }
  }
  void NullptrClear() {
    for (auto& hook : hooks_) { hook.NullptrClear(); }
  }
  void InsertAfter(ListHookArray* prev_skiplist_hook, int levels) {
    CHECK(empty());
    ListHook* prev_hook = &prev_skiplist_hook->hooks_[0];
    int i = 0;
    for (; i < levels; ++i, ++prev_hook) {
      while (prev_hook->nullptr_empty()) { prev_hook = (prev_hook - 1)->prev() + 1; }
      hooks_[i].InsertAfter(prev_hook);
    }
  }
  void Erase() {
    for (int i = 0; i < max_level; ++i) {
      if (hooks_[i].nullptr_empty()) { return; }
      hooks_[i].next()->AppendTo(hooks_[i].prev());
      hooks_[i].NullptrClear();
    }
  }
  static ListHookArray* ThisPtr4HookPtr(ListHook* slist_ptr, int level) {
    auto* hooks_ptr = (std::array<intrusive::ListHook, max_level>*)(slist_ptr - level);
    return OffsetStructField<self_type, decltype(hooks_), HooksOffset()>::StructPtr4FieldPtr(
        hooks_ptr);
  }
  void CheckEmpty() const {
    for (const auto& hook : hooks_) { CHECK(hook.empty()); }
  }
  void CheckNullptrEmpty() const {
    for (const auto& hook : hooks_) { CHECK(hook.nullptr_empty()); }
  }

  ListHook* mutable_hook(int i) { return &hooks_[i]; }

 private:
  template<typename Enabled = void>
  static constexpr int HooksOffset() {
    return offsetof(self_type, hooks_);
  }

  std::array<intrusive::ListHook, max_level> hooks_;
};

template<typename T, int N = 20>
struct SkipListHook {
 public:
  SkipListHook() : key_() { __Init__(); }
  using self_type = SkipListHook<T, N>;
  using hook_type = ListHookArray<N>;
  using key_type = T;
  static const int max_level = N;
  static_assert(N > 0, "invalid number of levels");
  template<typename Enabled = void>
  static constexpr int LevelZeroHookOffset() {
    return offsetof(SkipListHook, hook_) + hook_type::LevelZeroHookOffset();
  }

  bool empty() const { return hook_.empty(); }

  void __Init__() { hook_.NullptrClear(); }

  const T& key() const { return key_; }
  T* mut_key() { return &key_; }

  void CheckEmpty() const { return hook_.CheckNullptrEmpty(); }

  void Clear() {
    hook_.NullptrClear();
    mut_key()->__Delete__();
  }

  static self_type* Find(const key_type& key, hook_type* head, int size_shift) {
    ListHook* last_hook_less_than_key = SearchLastBottomHookLessThan(key, head, size_shift);
    if (last_hook_less_than_key->next() == head->mutable_hook(0)) { return nullptr; }
    self_type* searched = ThisPtr4HookPtr(last_hook_less_than_key->next(), 0);
    if (searched->key() == key) { return searched; }
    return nullptr;
  }
  static self_type* Erase(const key_type& key, hook_type* head, int size_shift) {
    self_type* searched = Find(key, head, size_shift);
    CHECK_NOTNULL(searched);
    Erase(searched);
    return searched;
  }
  static void Erase(self_type* elem) { elem->hook_.Erase(); }
  // return true if success
  static std::pair<self_type*, bool> Insert(self_type* elem, hook_type* head, int size_shift) {
    ListHook* prev_list_hook = SearchLastBottomHookLessThan(elem->key(), head, size_shift);
    self_type* maybe_searched = nullptr;
    if (prev_list_hook->next() == head->mutable_hook(0)) {
      maybe_searched = nullptr;
    } else {
      maybe_searched = ThisPtr4HookPtr(prev_list_hook->next(), 0);
    }
    self_type* ret_elem = nullptr;
    bool success = false;
    if (maybe_searched != nullptr && (maybe_searched->key() == elem->key())) {
      ret_elem = maybe_searched;
      success = false;
    } else {
      self_type* prev = ThisPtr4HookPtr(prev_list_hook, 0);
      ret_elem = elem;
      elem->hook_.InsertAfter(&prev->hook_, RandomNumLevels(size_shift));
      success = true;
    }
    // CHECK_EQ(Find(ret_elem->key(), head), ret_elem, GetMaxVal<int32_t>() / 2);
    return std::make_pair(ret_elem, success);
  }
  static SkipListHook* ThisPtr4HookPtr(ListHook* list_hook_ptr, int level) {
    auto* skip_list_ptr = hook_type::ThisPtr4HookPtr(list_hook_ptr, level);
    using FieldUtil = OffsetStructField<self_type, hook_type, SkipListIteratorOffset()>;
    return FieldUtil::StructPtr4FieldPtr(skip_list_ptr);
  }

 private:
  template<typename Enabled = void>
  static constexpr int SkipListIteratorOffset() {
    return offsetof(self_type, hook_);
  }
  static int32_t RandomNumLevels(int size_shift) {
    std::minstd_rand rand{std::random_device{}()};
    int32_t max_num_levels = std::min(size_shift, N);
    int32_t num_levels = 1;
    for (int i = 1; (rand() % 2 == 0) && i < max_num_levels; ++i) { ++num_levels; }
    return num_levels;
  }

  static ListHook* SearchLastBottomHookLessThan(const key_type& key, hook_type* head,
                                                int size_shift) {
    int max_num_level = std::min(size_shift, N);
    ListHook* list_hook = head->mutable_hook(max_num_level);
    for (int level = max_num_level - 1; level >= 0; --level) {
      --list_hook;
      while (list_hook->next() != head->mutable_hook(level)
             && ThisPtr4HookPtr(list_hook->next(), level)->key() < key) {
        list_hook = list_hook->next();
      }
    }
    return list_hook;
  }

  hook_type hook_;
  T key_;
};

template<typename ValueHookField>
class SkipListHead {
 public:
  SkipListHead() { __Init__(); }
  using value_type = typename ValueHookField::struct_type;
  using key_hook_type = typename ValueHookField::field_type;
  using key_type = typename key_hook_type::key_type;
  using value_key_level0_hook_struct_field =
      OffsetStructField<typename ValueHookField::field_type, intrusive::ListHook,
                        ValueHookField::field_type::LevelZeroHookOffset()>;
  using value_level0_hook_struct_field =
      ComposeStructField<ValueHookField, value_key_level0_hook_struct_field>;
  static const int max_level = key_hook_type::max_level;
  template<typename Enabled = void>
  static constexpr int IteratorHookOffset() {
    return offsetof(SkipListHead, skiplist_head_) + ListHookArray<max_level>::LevelZeroHookOffset();
  }

  void __Init__() {
    skiplist_head_.__Init__();
    size_ = 0;
  }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  value_type* Begin() {
    ListHook* head_level0 = skiplist_head_.mutable_hook(0);
    ListHook* begin_list_hook = head_level0->next();
    if (begin_list_hook == head_level0) { return nullptr; }
    return value_level0_hook_struct_field::StructPtr4FieldPtr(begin_list_hook);
  }

  value_type* Find(const key_type& key) {
    auto* key_hook_ptr = key_hook_type::Find(key, &skiplist_head_, size_shift());
    if (key_hook_ptr == nullptr) { return nullptr; }
    return ValueHookField::StructPtr4FieldPtr(key_hook_ptr);
  }
  const value_type* Find(const key_type& key) const {
    auto* key_hook_ptr = key_hook_type::Find(
        key, const_cast<ListHookArray<max_level>*>(&skiplist_head_), size_shift());
    if (key_hook_ptr == nullptr) { return nullptr; }
    return ValueHookField::StructPtr4FieldPtr(key_hook_ptr);
  }
  value_type* Erase(const key_type& key) {
    key_hook_type* erased = key_hook_type::Erase(key, &skiplist_head_, size_shift());
    --size_;
    return ValueHookField::StructPtr4FieldPtr(erased);
  }
  void Erase(value_type* elem) {
    key_hook_type::Erase(ValueHookField::FieldPtr4StructPtr(elem));
    --size_;
  }
  // return true if success
  std::pair<value_type*, bool> Insert(value_type* elem) {
    key_hook_type* elem_key_hook = ValueHookField::FieldPtr4StructPtr(elem);
    key_hook_type* ret_key_hook = nullptr;
    bool success = false;
    std::tie(ret_key_hook, success) =
        key_hook_type::Insert(elem_key_hook, &skiplist_head_, size_shift());
    if (success) { ++size_; }
    return std::make_pair(ValueHookField::StructPtr4FieldPtr(ret_key_hook), success);
  }

  template<typename Callback>
  void Clear(const Callback& cb) {
    using hook_type = ListHookArray<max_level>;
    for (; size_ > 0; --size_) {
      ListHook* begin_list_hook = skiplist_head_.mutable_hook(0)->next();
      auto* begin = hook_type::ThisPtr4HookPtr(begin_list_hook, 0);
      if (begin == &skiplist_head_) { break; }
      begin->Erase();
      cb(value_level0_hook_struct_field::StructPtr4FieldPtr(begin_list_hook));
    }
    CHECK(empty_debug());
  }
  void Clear() {
    Clear([](value_type*) {});
  }

  bool empty_debug() const {
    bool ret = (size_ == 0);
    if (ret) { skiplist_head_.CheckEmpty(); }
    return ret;
  }

 private:
  int size_shift() const { return std::log2(size_ + 1); }

  ListHookArray<max_level> skiplist_head_;
  volatile std::size_t size_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_EMBEDDED_SKIPLIST_H_
