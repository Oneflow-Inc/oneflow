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
#ifndef ONEFLOW_CORE_INTRUSIVE_LIST_H_
#define ONEFLOW_CORE_INTRUSIVE_LIST_H_

#include "oneflow/core/intrusive/ref.h"
#include "oneflow/core/intrusive/list_hook.h"

namespace oneflow {

namespace intrusive {

template<typename HookField>
class List {
 public:
  List(const List&) = delete;
  List(List&&) = delete;
  List() { this->__Init__(); }
  ~List() { this->Clear(); }

  using value_type = typename HookField::struct_type;
  using iterator_struct_field = HookField;

  template<typename Enabled = void>
  static constexpr int IteratorHookOffset() {
    return offsetof(List, list_head_) + intrusive::ListHead<HookField>::IteratorHookOffset();
  }

  std::size_t size() const { return list_head_.size(); }
  bool empty() const { return list_head_.empty(); }

  void CheckSize() const { list_head_.CheckSize(); }

  void __Init__() { list_head_.__Init__(); }

  value_type* Begin() {
    if (list_head_.empty()) { return nullptr; }
    return list_head_.Begin();
  }
  value_type* Prev(value_type* ptr) {
    if (ptr == nullptr) { return nullptr; }
    value_type* prev = list_head_.Prev(ptr);
    if (prev == list_head_.End()) { return nullptr; }
    return prev;
  }
  value_type* Next(value_type* ptr) {
    if (ptr == nullptr) { return nullptr; }
    value_type* next = list_head_.Next(ptr);
    if (next == list_head_.End()) { return nullptr; }
    return next;
  }
  value_type* Last() {
    if (list_head_.empty()) { return nullptr; }
    return list_head_.Last();
  }
  constexpr value_type* End() const { return nullptr; }

  void MoveToDstBack(value_type* ptr, List* dst) {
    list_head_.MoveToDstBack(ptr, &dst->list_head_);
  }
  void MoveToDstFront(value_type* ptr, List* dst) {
    list_head_.MoveToDstFront(ptr, &dst->list_head_);
  }
  value_type* MoveFrontToDstBack(List* dst) {
    value_type* begin = list_head_.Begin();
    MoveToDstBack(begin, dst);
    return begin;
  }
  value_type* MoveBackToDstBack(List* dst) {
    value_type* begin = list_head_.Last();
    MoveToDstBack(begin, dst);
    return begin;
  }

  void PushBack(value_type* ptr) {
    list_head_.PushBack(ptr);
    Ref::IncreaseRef(ptr);
  }

  void PushFront(value_type* ptr) {
    list_head_.PushFront(ptr);
    Ref::IncreaseRef(ptr);
  }

  void EmplaceBack(intrusive::shared_ptr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    ptr.__UnsafeMoveTo__(&raw_ptr);
    list_head_.PushBack(raw_ptr);
  }

  void EmplaceFront(intrusive::shared_ptr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    ptr.__UnsafeMoveTo__(&raw_ptr);
    list_head_.PushFront(raw_ptr);
  }

  intrusive::shared_ptr<value_type> Erase(value_type* ptr) {
    list_head_.Erase(ptr);
    return intrusive::shared_ptr<value_type>::__UnsafeMove__(ptr);
  }

  intrusive::shared_ptr<value_type> PopBack() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopBack(); }
    return intrusive::shared_ptr<value_type>::__UnsafeMove__(raw_ptr);
  }

  intrusive::shared_ptr<value_type> PopFront() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopFront(); }
    return intrusive::shared_ptr<value_type>::__UnsafeMove__(raw_ptr);
  }

  void MoveTo(List* list) { MoveToDstBack(list); }
  void MoveToDstBack(List* list) { list_head_.MoveToDstBack(&list->list_head_); }

  void Clear() {
    while (!empty()) { Ref::DecreaseRef(list_head_.PopFront()); }
  }

 private:
  intrusive::ListHead<HookField> list_head_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_LIST_H_
