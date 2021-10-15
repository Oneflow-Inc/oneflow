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
#ifndef ONEFLOW_CORE_INTRUSIVE_LIST_HOOK_H_
#define ONEFLOW_CORE_INTRUSIVE_LIST_HOOK_H_

#include "oneflow/core/intrusive/struct_traits.h"
#include <glog/logging.h>

namespace oneflow {

namespace intrusive {

struct ListHook {
 public:
  ListHook() { Clear(); }

  ListHook* prev() const { return prev_; }
  ListHook* next() const { return next_; }  // NOLINT

  void __Init__() { Clear(); }
  void Clear() {
    prev_ = this;
    next_ = this;
  }

  bool empty() const { return prev_ == this || next_ == this; }
  void AppendTo(ListHook* prev) {
    prev->set_next(this);
    this->set_prev(prev);
  }
  void InsertAfter(ListHook* prev) {
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
  void set_prev(ListHook* prev) { prev_ = prev; }
  void set_next(ListHook* next) { next_ = next; }

  ListHook* prev_;
  ListHook* next_;
};

#define LIST_HOOK_FOR_EACH(head_hook, elem_hook_struct_field, elem) \
  LIST_HOOK_FOR_EACH_WITH_EXPR(head_hook, elem_hook_struct_field, elem, 0)

#define LIST_HOOK_FOR_EACH_WITH_EXPR(head_hook, elem_hook_struct_field, elem, expr)   \
  for (typename elem_hook_struct_field::struct_type* elem = nullptr; elem == nullptr; \
       elem = nullptr, elem++)                                                        \
  LIST_HOOK_FOR_EACH_I(head_hook, __elem_hook__,                                      \
                       ((elem = elem_hook_struct_field::StructPtr4FieldPtr(__elem_hook__)), expr))

#define LIST_HOOK_FOR_EACH_I(head_hook, elem_hook, expr)                                     \
  for (intrusive::ListHook* __head_hook__ = (head_hook), *elem_hook = __head_hook__->next(), \
                            *__next_hook__ = elem_hook->next();                              \
       (elem_hook != __head_hook__) && ((expr) || true);                                     \
       elem_hook = __next_hook__, __next_hook__ = __next_hook__->next())

#define LIST_HOOK_UNSAFE_FOR_EACH(head_hook, elem_hook_struct_field, elem)            \
  for (typename elem_hook_struct_field::struct_type* elem = nullptr; elem == nullptr; \
       elem = nullptr, elem++)                                                        \
  LIST_HOOK_UNSAFE_FOR_EACH_I(head_hook, __elem_hook__,                               \
                              (elem = elem_hook_struct_field::StructPtr4FieldPtr(__elem_hook__)))

#define LIST_HOOK_UNSAFE_FOR_EACH_I(head_hook, elem_hook, expr)                              \
  for (intrusive::ListHook* __head_hook__ = (head_hook), *elem_hook = __head_hook__->next(); \
       (elem_hook != __head_hook__) && ((expr), true); elem_hook = elem_hook->next())

template<typename HookField>
class ListHead {
 public:
  ListHead() { Clear(); }
  using value_type = typename HookField::struct_type;
  static_assert(std::is_same<typename HookField::field_type, ListHook>::value, "no ListHook found");

  template<typename Enabled = void>
  static constexpr int IteratorHookOffset() {
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
    size_t hook_size = 0;
    for (ListHook* iter = container_.next(); iter != &container_; iter = iter->next()) {
      ++hook_size;
    }
    CHECK_EQ(size_, hook_size);
  }
  const value_type& Begin() const { return Next(End()); }
  const value_type& ReverseBegin() const { return Prev(End()); }
  const value_type& End() const { return *HookField::StructPtr4FieldPtr(&container()); }
  const value_type& Next(const value_type& current) const {
    return *HookField::StructPtr4FieldPtr(HookField::FieldPtr4StructPtr(&current)->next());
  }
  const value_type& Prev(const value_type& current) const {
    return *HookField::StructPtr4FieldPtr(HookField::FieldPtr4StructPtr(&current)->prev());
  }

  value_type* Begin() { return Next(End()); }
  value_type* Last() { return Prev(End()); }
  value_type* End() { return HookField::StructPtr4FieldPtr(mut_container()); }
  value_type* Next(value_type* current) {
    return HookField::StructPtr4FieldPtr(HookField::FieldPtr4StructPtr(current)->next());
  }
  value_type* Prev(value_type* current) {
    return HookField::StructPtr4FieldPtr(HookField::FieldPtr4StructPtr(current)->prev());
  }
  void __Init__() { Clear(); }

  void Clear() {
    container_.__Init__();
    size_ = 0;
  }

  void Erase(value_type* elem) {
    CHECK_GT(size_, 0);
    CHECK_NE(elem, End());
    ListHook* list_hook = HookField::FieldPtr4StructPtr(elem);
    CHECK(!list_hook->empty());
    list_hook->Erase();
    --size_;
  }
  void MoveToDstBack(value_type* elem, ListHead* dst) {
    CHECK(!container_.empty());
    auto* dst_rbegin = dst->container_.prev();
    auto* dst_end = &dst->container_;
    ListHook* elem_hook = HookField::FieldPtr4StructPtr(elem);
    elem_hook->next()->AppendTo(elem_hook->prev());
    elem_hook->AppendTo(dst_rbegin);
    dst_end->AppendTo(elem_hook);
    --size_;
    ++dst->size_;
  }
  void MoveToDstFront(value_type* elem, ListHead* dst) {
    CHECK(!container_.empty());
    auto* dst_end = &dst->container_;
    auto* dst_begin = dst->container_.next();
    ListHook* elem_hook = HookField::FieldPtr4StructPtr(elem);
    elem_hook->next()->AppendTo(elem_hook->prev());
    elem_hook->AppendTo(dst_end);
    dst_begin->AppendTo(elem_hook);
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
    ListHook* prev_list_hook = HookField::FieldPtr4StructPtr(prev_elem);
    ListHook* next_list_hook = prev_list_hook->next();
    ListHook* new_list_hook = HookField::FieldPtr4StructPtr(new_elem);
    CHECK(new_list_hook->empty());
    new_list_hook->AppendTo(prev_list_hook);
    next_list_hook->AppendTo(new_list_hook);
    ++size_;
  }
  const ListHook& container() const { return container_; }
  ListHook* mut_container() { return &container_; }

 private:
  ListHook container_;
  volatile std::size_t size_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_LIST_HOOK_H_
