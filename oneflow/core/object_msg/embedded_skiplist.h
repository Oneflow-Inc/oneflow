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
#ifndef ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_SKIPLIST_H_
#define ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_SKIPLIST_H_

#include <array>
#include <tuple>
#include <random>
#include <glog/logging.h>
#include "oneflow/core/object_msg/struct_traits.h"
#include "oneflow/core/object_msg/embedded_list.h"

namespace oneflow {

template<int max_level>
struct EmbeddedSkipListLink final {
 public:
  using self_type = EmbeddedSkipListLink<max_level>;
  template<typename Enabled = void>
  static constexpr int LevelZeroLinkOffset() {
    return 0;
  }

  bool empty() const { return links_[0].nullptr_empty(); }

  void __Init__() { Clear(); }
  void Clear() {
    for (auto& link : links_) { link.Clear(); }
  }
  void NullptrClear() {
    for (auto& link : links_) { link.NullptrClear(); }
  }
  void InsertAfter(EmbeddedSkipListLink* prev_skiplist_link, int levels) {
    CHECK(empty());
    EmbeddedListLink* prev_link = &prev_skiplist_link->links_[0];
    int i = 0;
    for (; i < levels; ++i, ++prev_link) {
      while (prev_link->nullptr_empty()) { prev_link = (prev_link - 1)->prev() + 1; }
      links_[i].InsertAfter(prev_link);
    }
  }
  void Erase() {
    for (int i = 0; i < max_level; ++i) {
      if (links_[i].nullptr_empty()) { return; }
      links_[i].next()->AppendTo(links_[i].prev());
      links_[i].NullptrClear();
    }
  }
  static EmbeddedSkipListLink* ThisPtr4LinkPtr(EmbeddedListLink* slist_ptr, int level) {
    auto* links_ptr = (std::array<EmbeddedListLink, max_level>*)(slist_ptr - level);
    return StructField<self_type, decltype(links_), LinksOffset()>::StructPtr4FieldPtr(links_ptr);
  }
  void CheckEmpty() const {
    for (const auto& link : links_) { CHECK(link.empty()); }
  }
  void CheckNullptrEmpty() const {
    for (const auto& link : links_) { CHECK(link.nullptr_empty()); }
  }

  EmbeddedListLink* mutable_link(int i) { return &links_[i]; }

 private:
  template<typename Enabled = void>
  static constexpr int LinksOffset() {
    return offsetof(self_type, links_);
  }

  std::array<EmbeddedListLink, max_level> links_;
};

template<typename T, int N = 20>
struct EmbeddedSkipListKey {
 public:
  using self_type = EmbeddedSkipListKey<T, N>;
  using link_type = EmbeddedSkipListLink<N>;
  using key_type = T;
  static const int max_level = N;
  static_assert(N > 0, "invalid number of levels");
  template<typename Enabled = void>
  static constexpr int LevelZeroLinkOffset() {
    return offsetof(EmbeddedSkipListKey, link_) + link_type::LevelZeroLinkOffset();
  }

  bool empty() const { return link_.empty(); }

  void __Init__() {
    link_.NullptrClear();
    KeyInitializer<std::is_scalar<T>::value>::Call(mut_key());
  }

  const T& key() const {
    const T* __attribute__((__may_alias__)) ptr = reinterpret_cast<const T*>(&key_buffer_[0]);
    return *ptr;
  }
  T* mut_key() {
    T* __attribute__((__may_alias__)) ptr = reinterpret_cast<T*>(&key_buffer_[0]);
    return ptr;
  }

  void CheckEmpty() const { return link_.CheckNullptrEmpty(); }

  void Clear() {
    link_.NullptrClear();
    mut_key()->__Delete__();
  }

  static self_type* Find(const key_type& key, link_type* head, int size_shift) {
    EmbeddedListLink* last_link_less_than_key = SearchLastBottomLinkLessThan(key, head, size_shift);
    if (last_link_less_than_key->next() == head->mutable_link(0)) { return nullptr; }
    self_type* searched = ThisPtr4LinkPtr(last_link_less_than_key->next(), 0);
    if (searched->key() == key) { return searched; }
    return nullptr;
  }
  static self_type* Erase(const key_type& key, link_type* head, int size_shift) {
    self_type* searched = Find(key, head, size_shift);
    CHECK_NOTNULL(searched);
    Erase(searched);
    return searched;
  }
  static void Erase(self_type* elem) { elem->link_.Erase(); }
  // return true if success
  static std::pair<self_type*, bool> Insert(self_type* elem, link_type* head, int size_shift) {
    EmbeddedListLink* prev_list_link = SearchLastBottomLinkLessThan(elem->key(), head, size_shift);
    self_type* maybe_searched = nullptr;
    if (prev_list_link->next() == head->mutable_link(0)) {
      maybe_searched = nullptr;
    } else {
      maybe_searched = ThisPtr4LinkPtr(prev_list_link->next(), 0);
    }
    self_type* ret_elem = nullptr;
    bool success = false;
    if (maybe_searched != nullptr && (maybe_searched->key() == elem->key())) {
      ret_elem = maybe_searched;
      success = false;
    } else {
      self_type* prev = ThisPtr4LinkPtr(prev_list_link, 0);
      ret_elem = elem;
      elem->link_.InsertAfter(&prev->link_, RandomNumLevels(size_shift));
      success = true;
    }
    // CHECK_EQ(Find(ret_elem->key(), head), ret_elem, GetMaxVal<int32_t>() / 2);
    return std::make_pair(ret_elem, success);
  }
  static EmbeddedSkipListKey* ThisPtr4LinkPtr(EmbeddedListLink* list_link_ptr, int level) {
    auto* skip_list_ptr = link_type::ThisPtr4LinkPtr(list_link_ptr, level);
    using FieldUtil = StructField<self_type, link_type, SkipListIteratorOffset()>;
    return FieldUtil::StructPtr4FieldPtr(skip_list_ptr);
  }

 private:
  template<bool is_scalar, typename Enabled = void>
  struct KeyInitializer {
    static void Call(T* key) {}
  };
  template<typename Enabled>
  struct KeyInitializer<false, Enabled> {
    static void Call(T* key) { key->__Init__(); }
  };

  template<typename Enabled = void>
  static constexpr int SkipListIteratorOffset() {
    return offsetof(self_type, link_);
  }
  static int32_t RandomNumLevels(int size_shift) {
    std::mt19937 rand{std::random_device{}()};
    int32_t max_num_levels = std::min(size_shift, N);
    int32_t num_levels = 1;
    for (int i = 1; (rand() % 2 == 0) && i < max_num_levels; ++i) { ++num_levels; }
    return num_levels;
  }

  static EmbeddedListLink* SearchLastBottomLinkLessThan(const key_type& key, link_type* head,
                                                        int size_shift) {
    int max_num_level = std::min(size_shift, N);
    EmbeddedListLink* list_link = head->mutable_link(max_num_level);
    for (int level = max_num_level - 1; level >= 0; --level) {
      --list_link;
      while (list_link->next() != head->mutable_link(level)
             && ThisPtr4LinkPtr(list_link->next(), level)->key() < key) {
        list_link = list_link->next();
      }
    }
    return list_link;
  }

  link_type link_;
  char key_buffer_[sizeof(T)];
};

template<typename ValueLinkField>
class EmbeddedSkipListHead {
 public:
  using value_type = typename ValueLinkField::struct_type;
  using key_link_type = typename ValueLinkField::field_type;
  using key_type = typename key_link_type::key_type;
  using value_key_level0_link_struct_field =
      StructField<typename ValueLinkField::field_type, EmbeddedListLink,
                  ValueLinkField::field_type::LevelZeroLinkOffset()>;
  using value_level0_link_struct_field =
      typename ComposeStructField<ValueLinkField, value_key_level0_link_struct_field>::type;
  static const int max_level = key_link_type::max_level;
  template<typename Enabled = void>
  static constexpr int ContainerLevelZeroLinkOffset() {
    return offsetof(EmbeddedSkipListHead, skiplist_head_)
           + EmbeddedSkipListLink<max_level>::LevelZeroLinkOffset();
  }

  void __Init__() {
    skiplist_head_.__Init__();
    size_ = 0;
  }

  std::size_t size() const { return size_; }
  bool empty() const { return size_ == 0; }

  value_type* Begin() {
    EmbeddedListLink* head_level0 = skiplist_head_.mutable_link(0);
    EmbeddedListLink* begin_list_link = head_level0->next();
    if (begin_list_link == head_level0) { return nullptr; }
    return value_level0_link_struct_field::StructPtr4FieldPtr(begin_list_link);
  }

  value_type* Find(const key_type& key) {
    auto* key_link_ptr = key_link_type::Find(key, &skiplist_head_, size_shift());
    if (key_link_ptr == nullptr) { return nullptr; }
    return ValueLinkField::StructPtr4FieldPtr(key_link_ptr);
  }
  const value_type* Find(const key_type& key) const {
    auto* key_link_ptr = key_link_type::Find(
        key, const_cast<EmbeddedSkipListLink<max_level>*>(&skiplist_head_), size_shift());
    if (key_link_ptr == nullptr) { return nullptr; }
    return ValueLinkField::StructPtr4FieldPtr(key_link_ptr);
  }
  value_type* Erase(const key_type& key) {
    key_link_type* erased = key_link_type::Erase(key, &skiplist_head_, size_shift());
    --size_;
    return ValueLinkField::StructPtr4FieldPtr(erased);
  }
  void Erase(value_type* elem) {
    key_link_type::Erase(ValueLinkField::FieldPtr4StructPtr(elem));
    --size_;
  }
  // return true if success
  std::pair<value_type*, bool> Insert(value_type* elem) {
    key_link_type* elem_key_link = ValueLinkField::FieldPtr4StructPtr(elem);
    key_link_type* ret_key_link = nullptr;
    bool success = false;
    std::tie(ret_key_link, success) =
        key_link_type::Insert(elem_key_link, &skiplist_head_, size_shift());
    if (success) { ++size_; }
    return std::make_pair(ValueLinkField::StructPtr4FieldPtr(ret_key_link), success);
  }

  template<typename Callback>
  void Clear(const Callback& cb) {
    using link_type = EmbeddedSkipListLink<max_level>;
    for (; size_ > 0; --size_) {
      EmbeddedListLink* begin_list_link = skiplist_head_.mutable_link(0)->next();
      auto* begin = link_type::ThisPtr4LinkPtr(begin_list_link, 0);
      if (begin == &skiplist_head_) { break; }
      begin->Erase();
      cb(value_level0_link_struct_field::StructPtr4FieldPtr(begin_list_link));
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

  EmbeddedSkipListLink<max_level> skiplist_head_;
  volatile std::size_t size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_SKIPLIST_H_
