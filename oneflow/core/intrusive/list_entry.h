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
#ifndef ONEFLOW_CORE_INTRUSIVE_LIST_ENTRY_H_
#define ONEFLOW_CORE_INTRUSIVE_LIST_ENTRY_H_

#include "oneflow/core/intrusive/struct_traits.h"
#include <glog/logging.h>

namespace oneflow {

namespace intrusive {

struct ListEntry {
 public:
  ListEntry() { Clear(); }

  ListEntry* prev() const { return prev_; }
  ListEntry* next() const { return next_; }  // NOLINT

  void __Init__() { Clear(); }
  void Clear() {
    prev_ = this;
    next_ = this;
  }

  bool empty() const { return prev_ == this || next_ == this; }
  void AppendTo(ListEntry* prev) {
    prev->set_next(this);
    this->set_prev(prev);
  }
  void InsertAfter(ListEntry* prev) {
    auto* next = prev->next();
    this->AppendTo(prev);
    next->AppendTo(this);
  }
  void Erase() {
    next_->AppendTo(prev_);
    Clear();
  }

  bool nullptr_empty() const { return prev_ == nullptr && next_ == nullptr; }

  void NullptrClear() {
    prev_ = nullptr;
    next_ = nullptr;
  }

 private:
  void set_prev(ListEntry* prev) { prev_ = prev; }
  void set_next(ListEntry* next) { next_ = next; }

  ListEntry* prev_;
  ListEntry* next_;
};

#define LIST_ENTRY_FOR_EACH(head_entry, elem_entry_struct_field, elem) \
  LIST_ENTRY_FOR_EACH_WITH_EXPR(head_entry, elem_entry_struct_field, elem, 0)

#define LIST_ENTRY_FOR_EACH_WITH_EXPR(head_entry, elem_entry_struct_field, elem, expr) \
  for (typename elem_entry_struct_field::struct_type* elem = nullptr; elem == nullptr; \
       elem = nullptr, elem++)                                                         \
  LIST_ENTRY_FOR_EACH_I(                                                               \
      head_entry, __elem_entry__,                                                      \
      ((elem = elem_entry_struct_field::StructPtr4FieldPtr(__elem_entry__)), expr))

#define LIST_ENTRY_FOR_EACH_I(head_entry, elem_entry, expr)                                       \
  for (intrusive::ListEntry* __head_entry__ = (head_entry), *elem_entry = __head_entry__->next(), \
                             *__next_entry__ = elem_entry->next();                                \
       (elem_entry != __head_entry__) && ((expr) || true);                                        \
       elem_entry = __next_entry__, __next_entry__ = __next_entry__->next())

#define LIST_ENTRY_UNSAFE_FOR_EACH(head_entry, elem_entry_struct_field, elem)          \
  for (typename elem_entry_struct_field::struct_type* elem = nullptr; elem == nullptr; \
       elem = nullptr, elem++)                                                         \
  LIST_ENTRY_UNSAFE_FOR_EACH_I(                                                        \
      head_entry, __elem_entry__,                                                      \
      (elem = elem_entry_struct_field::StructPtr4FieldPtr(__elem_entry__)))

#define LIST_ENTRY_UNSAFE_FOR_EACH_I(head_entry, elem_entry, expr)                                \
  for (intrusive::ListEntry* __head_entry__ = (head_entry), *elem_entry = __head_entry__->next(); \
       (elem_entry != __head_entry__) && ((expr), true); elem_entry = elem_entry->next())

template<typename EntryField>
class ListHead {
 public:
  ListHead() { Clear(); }
  using value_type = typename EntryField::struct_type;
  static_assert(std::is_same<typename EntryField::field_type, ListEntry>::value,
                "no ListEntry found");

  template<typename Enabled = void>
  static constexpr int IteratorEntryOffset() {
    return offsetof(ListHead, container_);
  }

  std::size_t size() const { return size_; }
  bool empty() const {
    bool list_empty = (&Begin() == &End());
    bool size_empty = (size_ == 0);
    CHECK_EQ(list_empty, size_empty);
    return size_empty;
  }
  void CheckSize() const {
    size_t entry_size = 0;
    for (ListEntry* iter = container_.next(); iter != &container_; iter = iter->next()) {
      ++entry_size;
    }
    CHECK_EQ(size_, entry_size);
  }
  const value_type& Begin() const { return Next(End()); }
  const value_type& ReverseBegin() const { return Prev(End()); }
  const value_type& End() const { return *EntryField::StructPtr4FieldPtr(&container()); }
  const value_type& Next(const value_type& current) const {
    return *EntryField::StructPtr4FieldPtr(EntryField::FieldPtr4StructPtr(&current)->next());
  }
  const value_type& Prev(const value_type& current) const {
    return *EntryField::StructPtr4FieldPtr(EntryField::FieldPtr4StructPtr(&current)->prev());
  }

  value_type* Begin() { return Next(End()); }
  value_type* Last() { return Prev(End()); }
  value_type* End() { return EntryField::StructPtr4FieldPtr(mut_container()); }
  value_type* Next(value_type* current) {
    return EntryField::StructPtr4FieldPtr(EntryField::FieldPtr4StructPtr(current)->next());
  }
  value_type* Prev(value_type* current) {
    return EntryField::StructPtr4FieldPtr(EntryField::FieldPtr4StructPtr(current)->prev());
  }
  void __Init__() { Clear(); }

  void Clear() {
    container_.__Init__();
    size_ = 0;
  }

  void Erase(value_type* elem) {
    CHECK_GT(size_, 0);
    CHECK_NE(elem, End());
    ListEntry* list_entry = EntryField::FieldPtr4StructPtr(elem);
    CHECK(!list_entry->empty());
    list_entry->Erase();
    --size_;
  }
  void MoveToDstBack(value_type* elem, ListHead* dst) {
    CHECK(!container_.empty());
    auto* dst_rbegin = dst->container_.prev();
    auto* dst_end = &dst->container_;
    ListEntry* elem_entry = EntryField::FieldPtr4StructPtr(elem);
    elem_entry->next()->AppendTo(elem_entry->prev());
    elem_entry->AppendTo(dst_rbegin);
    dst_end->AppendTo(elem_entry);
    --size_;
    ++dst->size_;
  }
  void MoveToDstFront(value_type* elem, ListHead* dst) {
    CHECK(!container_.empty());
    auto* dst_end = &dst->container_;
    auto* dst_begin = dst->container_.next();
    ListEntry* elem_entry = EntryField::FieldPtr4StructPtr(elem);
    elem_entry->next()->AppendTo(elem_entry->prev());
    elem_entry->AppendTo(dst_end);
    dst_begin->AppendTo(elem_entry);
    --size_;
    ++dst->size_;
  }
  void PushBack(value_type* elem) { InsertAfter(Last(), elem); }
  void PushFront(value_type* elem) { InsertAfter(End(), elem); }
  value_type* PopBack() {
    CHECK(!empty());
    value_type* last = Last();
    Erase(last);
    return last;
  }
  value_type* PopFront() {
    CHECK(!empty());
    value_type* first = Begin();
    Erase(first);
    return first;
  }
  void MoveToDstBack(ListHead* dst) {
    if (container_.empty()) { return; }
    auto* dst_last = dst->container_.prev();
    auto* dst_end = &dst->container_;
    auto* this_first = container_.next();
    auto* this_last = container_.prev();
    this_first->AppendTo(dst_last);
    dst_end->AppendTo(this_last);
    dst->size_ += size();
    this->Clear();
  }

 private:
  void InsertAfter(value_type* prev_elem, value_type* new_elem) {
    ListEntry* prev_list_entry = EntryField::FieldPtr4StructPtr(prev_elem);
    ListEntry* next_list_entry = prev_list_entry->next();
    ListEntry* new_list_entry = EntryField::FieldPtr4StructPtr(new_elem);
    CHECK(new_list_entry->empty());
    new_list_entry->AppendTo(prev_list_entry);
    next_list_entry->AppendTo(new_list_entry);
    ++size_;
  }
  const ListEntry& container() const { return container_; }
  ListEntry* mut_container() { return &container_; }

 private:
  ListEntry container_;
  volatile std::size_t size_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_LIST_ENTRY_H_
