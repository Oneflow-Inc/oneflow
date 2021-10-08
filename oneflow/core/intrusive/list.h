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

#include <typeinfo>
#include "oneflow/core/intrusive/intrusive_core.h"
#include "oneflow/core/intrusive/list_entry.h"
#include "oneflow/core/intrusive/struct_traits.h"

namespace oneflow {

namespace intrusive {

template<typename ValueLinkField>
class List {
 public:
  static_assert(std::is_same<typename ValueLinkField::field_type, intrusive::ListEntry>::value, "");
  List(const List&) = delete;
  List(List&&) = delete;
  List() { this->__Init__(); }
  ~List() { this->Clear(); }

  using value_type = typename ValueLinkField::struct_type;
  using iterator_struct_field = ValueLinkField;

  template<typename Enabled = void>
  static constexpr int IteratorEntryOffset() {
    return offsetof(List, list_head_) + intrusive::ListHead<ValueLinkField>::IteratorEntryOffset();
  }

  std::size_t size() const { return list_head_.size(); }
  bool empty() const { return list_head_.empty(); }

  void CheckSize() const { list_head_.CheckSize(); }

  void __Init__() { list_head_.__Init__(); }

  value_type* Begin() {
    if (list_head_.empty()) { return nullptr; }
    return list_head_.Begin();
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
    PtrUtil::Ref(ptr);
  }

  void PushFront(value_type* ptr) {
    list_head_.PushFront(ptr);
    PtrUtil::Ref(ptr);
  }

  void EmplaceBack(intrusive::SharedPtr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    ptr.__UnsafeMoveTo__(&raw_ptr);
    list_head_.PushBack(raw_ptr);
  }

  void EmplaceFront(intrusive::SharedPtr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    ptr.__UnsafeMoveTo__(&raw_ptr);
    list_head_.PushFront(raw_ptr);
  }

  intrusive::SharedPtr<value_type> Erase(value_type* ptr) {
    list_head_.Erase(ptr);
    return intrusive::SharedPtr<value_type>::__UnsafeMove__(ptr);
  }

  intrusive::SharedPtr<value_type> PopBack() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopBack(); }
    return intrusive::SharedPtr<value_type>::__UnsafeMove__(raw_ptr);
  }

  intrusive::SharedPtr<value_type> PopFront() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopFront(); }
    return intrusive::SharedPtr<value_type>::__UnsafeMove__(raw_ptr);
  }

  void MoveTo(List* list) { MoveToDstBack(list); }
  void MoveToDstBack(List* list) { list_head_.MoveToDstBack(&list->list_head_); }

  void Clear() {
    while (!empty()) { PtrUtil::ReleaseRef(list_head_.PopFront()); }
  }

 private:
  intrusive::ListHead<ValueLinkField> list_head_;
};

template<typename ValueLinkField, int field_counter>
class HeadFreeList {
 public:
  static_assert(std::is_same<typename ValueLinkField::field_type, intrusive::ListEntry>::value, "");
  HeadFreeList(const HeadFreeList&) = delete;
  HeadFreeList(HeadFreeList&&) = delete;
  HeadFreeList() { this->__Init__(); }
  ~HeadFreeList() { this->Clear(); }

  using value_type = typename ValueLinkField::struct_type;
  using iterator_struct_field = ValueLinkField;

  // field_counter is last field_number
  static const int field_number_in_countainter = field_counter + 1;

  template<typename Enabled = void>
  static constexpr int IteratorEntryOffset() {
    return offsetof(HeadFreeList, list_head_)
           + intrusive::ListHead<ValueLinkField>::IteratorEntryOffset();
  }

  std::size_t size() const { return list_head_.size(); }
  bool empty() const { return list_head_.empty(); }

  void __Init__() {
    list_head_.__Init__();
    static_assert(
        std::is_same<HeadFreeList,
                     INTRUSIVE_FIELD_TYPE(typename value_type, field_number_in_countainter)>::value,
        "");
    using ThisInContainer =
        StructField<value_type, HeadFreeList,
                    INTRUSIVE_FIELD_OFFSET(value_type, field_number_in_countainter)>;
    container_ = ThisInContainer::StructPtr4FieldPtr(this);
  }

  value_type* Begin() {
    if (list_head_.empty()) { return nullptr; }
    return list_head_.Begin();
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

  void MoveToDstBack(value_type* ptr, HeadFreeList* dst) {
    list_head_.MoveToDstBack(ptr, &dst->list_head_);
    MoveReference(ptr, dst);
  }
  void MoveToDstFront(value_type* ptr, HeadFreeList* dst) {
    list_head_.MoveToDstFront(ptr, &dst->list_head_);
    MoveReference(ptr, dst);
  }
  value_type* MoveFrontToDstBack(HeadFreeList* dst) {
    value_type* begin = list_head_.Begin();
    MoveToDstBack(begin, dst);
    return begin;
  }
  value_type* MoveBackToDstBack(HeadFreeList* dst) {
    value_type* begin = list_head_.Last();
    MoveToDstBack(begin, dst);
    return begin;
  }

  void PushBack(value_type* ptr) {
    list_head_.PushBack(ptr);
    if (container_ != ptr) { PtrUtil::Ref(ptr); }
  }

  void PushFront(value_type* ptr) {
    list_head_.PushFront(ptr);
    if (container_ != ptr) { PtrUtil::Ref(ptr); }
  }

  void EmplaceBack(intrusive::SharedPtr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    if (container_ != ptr.Mutable()) {
      ptr.__UnsafeMoveTo__(&raw_ptr);
    } else {
      raw_ptr = ptr.Mutable();
    }
    list_head_.PushBack(raw_ptr);
  }

  void EmplaceFront(intrusive::SharedPtr<value_type>&& ptr) {
    value_type* raw_ptr = nullptr;
    if (container_ != ptr.Mutable()) {
      ptr.__UnsafeMoveTo__(&raw_ptr);
    } else {
      raw_ptr = ptr.Mutable();
    }
    list_head_.PushFront(raw_ptr);
  }

  intrusive::SharedPtr<value_type> Erase(value_type* ptr) {
    list_head_.Erase(ptr);
    if (container_ != ptr) {
      return intrusive::SharedPtr<value_type>::__UnsafeMove__(ptr);
    } else {
      return intrusive::SharedPtr<value_type>(ptr);
    }
  }

  intrusive::SharedPtr<value_type> PopBack() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopBack(); }
    if (container_ != raw_ptr) {
      return intrusive::SharedPtr<value_type>::__UnsafeMove__(raw_ptr);
    } else {
      return intrusive::SharedPtr<value_type>(raw_ptr);
    }
  }

  intrusive::SharedPtr<value_type> PopFront() {
    value_type* raw_ptr = nullptr;
    if (!list_head_.empty()) { raw_ptr = list_head_.PopFront(); }
    if (container_ != raw_ptr) {
      return intrusive::SharedPtr<value_type>::__UnsafeMove__(raw_ptr);
    } else {
      return intrusive::SharedPtr<value_type>(raw_ptr);
    }
  }

  void MoveTo(HeadFreeList* list) { MoveToDstBack(list); }
  void MoveToDstBack(HeadFreeList* list) {
    while (!empty()) { MoveToDstBack(list_head_.Begin(), list); }
  }

  void Clear() {
    while (!empty()) {
      auto* ptr = list_head_.PopFront();
      if (container_ != ptr) { PtrUtil::ReleaseRef(ptr); }
    }
  }

 private:
  void MoveReference(value_type* ptr, HeadFreeList* dst) {
    if (ptr == container_ && ptr != dst->container_) {
      PtrUtil::Ref(ptr);
    } else if (ptr != container_ && ptr == dst->container_) {
      PtrUtil::ReleaseRef(ptr);
    } else {
      // do nothing
    }
  }

  intrusive::ListHead<ValueLinkField> list_head_;
  const value_type* container_;
};

}  // namespace intrusive

}  // namespace oneflow

#endif  // ONEFLOW_CORE_INTRUSIVE_LIST_H_
