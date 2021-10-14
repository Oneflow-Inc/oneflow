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
#include "oneflow/core/intrusive/list_entry.h"

namespace oneflow {

namespace intrusive {

template<int max_level>
struct ListEntryArray final {
 public:
  ListEntryArray() { Clear(); }
  using self_type = ListEntryArray<max_level>;
  template<typename Enabled = void>
  static constexpr int LevelZeroEntryOffset() {
    return 0;
  }

  bool empty() const { return entries_[0].nullptr_empty(); }

  void __Init__() { Clear(); }

  void Clear() {
    for (auto& entry : entries_) { entry.Clear(); }
  }
  void NullptrClear() {
    for (auto& entry : entries_) { entry.NullptrClear(); }
  }
  void InsertAfter(ListEntryArray* prev_skiplist_entry, int levels) {
    CHECK(empty());
    ListEntry* prev_entry = &prev_skiplist_entry->entries_[0];
    int i = 0;
    for (; i < levels; ++i, ++prev_entry) {
      while (prev_entry->nullptr_empty()) { prev_entry = (prev_entry - 1)->prev() + 1; }
      entries_[i].InsertAfter(prev_entry);
    }
  }
  void Erase() {
    for (int i = 0; i < max_level; ++i) {
      if (entries_[i].nullptr_empty()) { return; }
      entries_[i].next()->AppendTo(entries_[i].prev());
      entries_[i].NullptrClear();
    }
  }
  static ListEntryArray* ThisPtr4EntryPtr(ListEntry* slist_ptr, int level) {
    auto* entries_ptr = (std::array<intrusive::ListEntry, max_level>*)(slist_ptr - level);
    return StructField<self_type, decltype(entries_), EntrysOffset()>::StructPtr4FieldPtr(
        entries_ptr);
  }
  void CheckEmpty() const {
    for (const auto& entry : entries_) { CHECK(entry.empty()); }
  }
  void CheckNullptrEmpty() const {
    for (const auto& entry : entries_) { CHECK(entry.nullptr_empty()); }
  }

  ListEntry* mutable_entry(int i) { return &entries_[i]; }

 private:
  template<typename Enabled = void>
  static constexpr int EntrysOffset() {
    return offsetof(self_type, entries_);
  }

  std::array<intrusive::ListEntry, max_level> entries_;
};

template<typename T, int N = 20>
struct SkipListEntry {
 public:
  SkipListEntry() : key_() { __Init__(); }
  using self_type = SkipListEntry<T, N>;
  using entry_type = ListEntryArray<N>;
  using key_type = T;
  static const int max_level = N;
  static_assert(N > 0, "invalid number of levels");
  template<typename Enabled = void>
  static constexpr int LevelZeroEntryOffset() {
    return offsetof(SkipListEntry, entry_) + entry_type::LevelZeroEntryOffset();
  }

  bool empty() const { return entry_.empty(); }

  void __Init__() { entry_.NullptrClear(); }

  const T& key() const { return key_; }
  T* mut_key() { return &key_; }

  void CheckEmpty() const { return entry_.CheckNullptrEmpty(); }

  void Clear() {
    entry_.NullptrClear();
    mut_key()->__Delete__();
  }

  static self_type* Find(const key_type& key, entry_type* head, int size_shift) {
    ListEntry* last_entry_less_than_key = SearchLastBottomEntryLessThan(key, head, size_shift);
    if (last_entry_less_than_key->next() == head->mutable_entry(0)) { return nullptr; }
    self_type* searched = ThisPtr4EntryPtr(last_entry_less_than_key->next(), 0);
    if (searched->key() == key) { return searched; }
    return nullptr;
  }
  static self_type* Erase(const key_type& key, entry_type* head, int size_shift) {
    self_type* searched = Find(key, head, size_shift);
    CHECK_NOTNULL(searched);
    Erase(searched);
    return searched;
  }
  static void Erase(self_type* elem) { elem->entry_.Erase(); }
  // return true if success
  static std::pair<self_type*, bool> Insert(self_type* elem, entry_type* head, int size_shift) {
    ListEntry* prev_list_entry = SearchLastBottomEntryLessThan(elem->key(), head, size_shift);
    self_type* maybe_searched = nullptr;
    if (prev_list_entry->next() == head->mutable_entry(0)) {
      maybe_searched = nullptr;
    } else {
      maybe_searched = ThisPtr4EntryPtr(prev_list_entry->next(), 0);
    }
    self_type* ret_elem = nullptr;
    bool success = false;
    if (maybe_searched != nullptr && (maybe_searched->key() == elem->key())) {
      ret_elem = maybe_searched;
      success = false;
    } else {
      self_type* prev = ThisPtr4EntryPtr(prev_list_entry, 0);
      ret_elem = elem;
      elem->entry_.InsertAfter(&prev->entry_, RandomNumLevels(size_shift));
      success = true;
    }
    // CHECK_EQ(Find(ret_elem->key(), head), ret_elem, GetMaxVal<int32_t>() / 2);
    return std::make_pair(ret_elem, success);
  }
  static SkipListEntry* ThisPtr4EntryPtr(ListEntry* list_entry_ptr, int level) {
    auto* skip_list_ptr = entry_type::ThisPtr4EntryPtr(list_entry_ptr, level);
    using FieldUtil = StructField<self_type, entry_type, SkipListIteratorOffset()>;
    return FieldUtil::StructPtr4FieldPtr(skip_list_ptr);
  }

 private:
  template<typename Enabled = void>
  static constexpr int SkipListIteratorOffset() {
    return offsetof(self_type, entry_);
  }
  static int32_t RandomNumLevels(int size_shift) {
    std::minstd_rand rand{std::random_device{}()};
    int32_t max_num_levels = std::min(size_shift, N);
    int32_t num_levels = 1;
    for (int i = 1; (rand() % 2 == 0) && i < max_num_levels; ++i) { ++num_levels; }
    return num_levels;
  }

  static ListEntry* SearchLastBottomEntryLessThan(const key_type& key, entry_type* head,
                                                  int size_shift) {
    int max_num_level = std::min(size_shift, N);
    ListEntry* list_entry = head->mutable_entry(max_num_level);
    for (int level = max_num_level - 1; level >= 0; --level) {
      --list_entry;
      while (list_entry->next() != head->mutable_entry(level)
             && ThisPtr4EntryPtr(list_entry->next(), level)->key() < key) {
        list_entry = list_entry->next();
      }
    }
    return list_entry;
  }

  entry_type entry_;
  T key_;
};

template<typename ValueEntryField>
class SkipListHead {
 public:
  SkipListHead() { __Init__(); }
  using value_type = typename ValueEntryField::struct_type;
  using key_entry_type = typename ValueEntryField::field_type;
  using key_type = typename key_entry_type::key_type;
  using value_key_level0_entry_struct_field =
      StructField<typename ValueEntryField::field_type, intrusive::ListEntry,
                  ValueEntryField::field_type::LevelZeroEntryOffset()>;
  using value_level0_entry_struct_field =
      typename ComposeStructField<ValueEntryField, value_key_level0_entry_struct_field>::type;
  static const int max_level = key_entry_type::max_level;
  template<typename Enabled = void>
  static constexpr int IteratorEntryOffset() {
    return offsetof(SkipListHead, skiplist_head_)
           + ListEntryArray<max_level>::LevelZeroEntryOffset();
  }

  void __Init__() {
    skiplist_head_.__Init__();
    size_ = 0;
  }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  value_type* Begin() {
    ListEntry* head_level0 = skiplist_head_.mutable_entry(0);
    ListEntry* begin_list_entry = head_level0->next();
    if (begin_list_entry == head_level0) { return nullptr; }
    return value_level0_entry_struct_field::StructPtr4FieldPtr(begin_list_entry);
  }

  value_type* Find(const key_type& key) {
    auto* key_entry_ptr = key_entry_type::Find(key, &skiplist_head_, size_shift());
    if (key_entry_ptr == nullptr) { return nullptr; }
    return ValueEntryField::StructPtr4FieldPtr(key_entry_ptr);
  }
  const value_type* Find(const key_type& key) const {
    auto* key_entry_ptr = key_entry_type::Find(
        key, const_cast<ListEntryArray<max_level>*>(&skiplist_head_), size_shift());
    if (key_entry_ptr == nullptr) { return nullptr; }
    return ValueEntryField::StructPtr4FieldPtr(key_entry_ptr);
  }
  value_type* Erase(const key_type& key) {
    key_entry_type* erased = key_entry_type::Erase(key, &skiplist_head_, size_shift());
    --size_;
    return ValueEntryField::StructPtr4FieldPtr(erased);
  }
  void Erase(value_type* elem) {
    key_entry_type::Erase(ValueEntryField::FieldPtr4StructPtr(elem));
    --size_;
  }
  // return true if success
  std::pair<value_type*, bool> Insert(value_type* elem) {
    key_entry_type* elem_key_entry = ValueEntryField::FieldPtr4StructPtr(elem);
    key_entry_type* ret_key_entry = nullptr;
    bool success = false;
    std::tie(ret_key_entry, success) =
        key_entry_type::Insert(elem_key_entry, &skiplist_head_, size_shift());
    if (success) { ++size_; }
    return std::make_pair(ValueEntryField::StructPtr4FieldPtr(ret_key_entry), success);
  }

  template<typename Callback>
  void Clear(const Callback& cb) {
    using entry_type = ListEntryArray<max_level>;
    for (; size_ > 0; --size_) {
      ListEntry* begin_list_entry = skiplist_head_.mutable_entry(0)->next();
      auto* begin = entry_type::ThisPtr4EntryPtr(begin_list_entry, 0);
      if (begin == &skiplist_head_) { break; }
      begin->Erase();
      cb(value_level0_entry_struct_field::StructPtr4FieldPtr(begin_list_entry));
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

  ListEntryArray<max_level> skiplist_head_;
  volatile std::size_t size_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_EMBEDDED_SKIPLIST_H_
