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
#ifndef ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_LIST_H_
#define ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_LIST_H_

#include "oneflow/core/object_msg/struct_traits.h"
#include <glog/logging.h>

namespace oneflow {

struct EmbeddedListLink {
 public:
  EmbeddedListLink* prev() const { return prev_; }
  EmbeddedListLink* next() const { return next_; }

  void __Init__() { Clear(); }
  void Clear() {
    prev_ = this;
    next_ = this;
  }

  bool empty() const { return prev_ == this || next_ == this; }
  void AppendTo(EmbeddedListLink* prev) {
    prev->set_next(this);
    this->set_prev(prev);
  }
  void InsertAfter(EmbeddedListLink* prev) {
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
  void set_prev(EmbeddedListLink* prev) { prev_ = prev; }
  void set_next(EmbeddedListLink* next) { next_ = next; }

  EmbeddedListLink* prev_;
  EmbeddedListLink* next_;
};

#define EMBEDDED_LIST_FOR_EACH(head_link, elem_link_struct_field, elem) \
  EMBEDDED_LIST_FOR_EACH_WITH_EXPR(head_link, elem_link_struct_field, elem, 0)

#define EMBEDDED_LIST_FOR_EACH_WITH_EXPR(head_link, elem_link_struct_field, elem, expr) \
  for (typename elem_link_struct_field::struct_type* elem = nullptr; elem == nullptr;   \
       elem = nullptr, elem++)                                                          \
  EMBEDDED_LIST_FOR_EACH_I(                                                             \
      head_link, __elem_link__,                                                         \
      ((elem = elem_link_struct_field::StructPtr4FieldPtr(__elem_link__)), expr))

#define EMBEDDED_LIST_FOR_EACH_I(head_link, elem_link, expr)                              \
  for (EmbeddedListLink* __head_link__ = (head_link), *elem_link = __head_link__->next(), \
                         *__next_link__ = elem_link->next();                              \
       (elem_link != __head_link__) && ((expr) || true);                                  \
       elem_link = __next_link__, __next_link__ = __next_link__->next())

#define EMBEDDED_LIST_UNSAFE_FOR_EACH(head_link, elem_link_struct_field, elem)        \
  for (typename elem_link_struct_field::struct_type* elem = nullptr; elem == nullptr; \
       elem = nullptr, elem++)                                                        \
  EMBEDDED_LIST_UNSAFE_FOR_EACH_I(                                                    \
      head_link, __elem_link__,                                                       \
      (elem = elem_link_struct_field::StructPtr4FieldPtr(__elem_link__)))

#define EMBEDDED_LIST_UNSAFE_FOR_EACH_I(head_link, elem_link, expr)                       \
  for (EmbeddedListLink* __head_link__ = (head_link), *elem_link = __head_link__->next(); \
       (elem_link != __head_link__) && ((expr), true); elem_link = elem_link->next())

template<typename LinkField>
class EmbeddedListHead {
 public:
  using value_type = typename LinkField::struct_type;
  static_assert(std::is_same<typename LinkField::field_type, EmbeddedListLink>::value,
                "no EmbeddedListLink found");

  template<typename Enabled = void>
  static constexpr int ContainerLinkOffset() {
    return offsetof(EmbeddedListHead, container_);
  }

  std::size_t size() const { return size_; }
  bool empty() const {
    bool list_empty = (&Begin() == &End());
    bool size_empty = (size_ == 0);
    CHECK_EQ(list_empty, size_empty);
    return size_empty;
  }
  void CheckSize() const {
    size_t link_size = 0;
    for (EmbeddedListLink* iter = container_.next(); iter != &container_; iter = iter->next()) {
      ++link_size;
    }
    CHECK_EQ(size_, link_size);
  }
  const value_type& Begin() const { return Next(End()); }
  const value_type& ReverseBegin() const { return Prev(End()); }
  const value_type& End() const { return *LinkField::StructPtr4FieldPtr(&container()); }
  const value_type& Next(const value_type& current) const {
    return *LinkField::StructPtr4FieldPtr(LinkField::FieldPtr4StructPtr(&current)->next());
  }
  const value_type& Prev(const value_type& current) const {
    return *LinkField::StructPtr4FieldPtr(LinkField::FieldPtr4StructPtr(&current)->prev());
  }

  value_type* Begin() { return Next(End()); }
  value_type* Last() { return Prev(End()); }
  value_type* End() { return LinkField::StructPtr4FieldPtr(mut_container()); }
  value_type* Next(value_type* current) {
    return LinkField::StructPtr4FieldPtr(LinkField::FieldPtr4StructPtr(current)->next());
  }
  value_type* Prev(value_type* current) {
    return LinkField::StructPtr4FieldPtr(LinkField::FieldPtr4StructPtr(current)->prev());
  }
  void __Init__() { Clear(); }

  void Clear() {
    container_.__Init__();
    size_ = 0;
  }

  void Erase(value_type* elem) {
    CHECK_GT(size_, 0);
    CHECK_NE(elem, End());
    EmbeddedListLink* list_link = LinkField::FieldPtr4StructPtr(elem);
    CHECK(!list_link->empty());
    list_link->Erase();
    --size_;
  }
  void MoveToDstBack(value_type* elem, EmbeddedListHead* dst) {
    CHECK(!container_.empty());
    auto* dst_rbegin = dst->container_.prev();
    auto* dst_end = &dst->container_;
    EmbeddedListLink* elem_link = LinkField::FieldPtr4StructPtr(elem);
    elem_link->next()->AppendTo(elem_link->prev());
    elem_link->AppendTo(dst_rbegin);
    dst_end->AppendTo(elem_link);
    --size_;
    ++dst->size_;
  }
  void MoveToDstFront(value_type* elem, EmbeddedListHead* dst) {
    CHECK(!container_.empty());
    auto* dst_end = &dst->container_;
    auto* dst_begin = dst->container_.next();
    EmbeddedListLink* elem_link = LinkField::FieldPtr4StructPtr(elem);
    elem_link->next()->AppendTo(elem_link->prev());
    elem_link->AppendTo(dst_end);
    dst_begin->AppendTo(elem_link);
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
  void MoveToDstBack(EmbeddedListHead* dst) {
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
    EmbeddedListLink* prev_list_link = LinkField::FieldPtr4StructPtr(prev_elem);
    EmbeddedListLink* next_list_link = prev_list_link->next();
    EmbeddedListLink* new_list_link = LinkField::FieldPtr4StructPtr(new_elem);
    CHECK(new_list_link->empty());
    new_list_link->AppendTo(prev_list_link);
    next_list_link->AppendTo(new_list_link);
    ++size_;
  }
  const EmbeddedListLink& container() const { return container_; }
  EmbeddedListLink* mut_container() { return &container_; }

 private:
  EmbeddedListLink container_;
  volatile std::size_t size_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_LIST_H_
