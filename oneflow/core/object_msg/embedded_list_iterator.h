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
#ifndef ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_LIST_ITERATOR_H_
#define ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_LIST_ITERATOR_H_

#include "oneflow/core/object_msg/embedded_list.h"

namespace oneflow {

template<bool IsConst, typename T1, typename T2>
struct if_c {
  typedef T1 type;
};

template<typename T1, typename T2>
struct if_c<false, T1, T2> {
  typedef T2 type;
};

template<typename ValueType, bool IsConst>
struct if_const_ref {
  typedef ValueType& type;
};

template<typename ValueType>
struct if_const_ref<ValueType, true> {
  typedef const ValueType& type;
};

template<typename ValueType, bool IsConst>
struct if_const_pointer {
  typedef ValueType* type;
};

template<typename ValueType>
struct if_const_pointer<ValueType, true> {
  typedef const ValueType* type;
};

template<typename LinkField, bool IsConst>
class embedded_list_iterator {
 public:
  using iterator = embedded_list_iterator<LinkField, IsConst>;
  using value_type = typename LinkField::struct_type;
  using field_type = typename LinkField::field_type;

  using pointer = typename if_const_pointer<value_type, IsConst>::type;
  using reference = typename if_const_ref<value_type, IsConst>::type;
  class nat;
  using unconst_iterator =
      typename if_c<IsConst, embedded_list_iterator<LinkField, false>, nat>::type;

  embedded_list_iterator() : container_(nullptr) {}
  explicit embedded_list_iterator(field_type* container = nullptr) : container_(container) {}
  embedded_list_iterator(const embedded_list_iterator& other) : container_(other.container_) {}
  embedded_list_iterator(const unconst_iterator& other) : container_(other.container_) {}

  iterator& operator=(const iterator& other) {
    container_ = other.container_;
    return *this;
  }

  iterator& operator=(const field_type& container) {
    container_ = &container;
    return *this;
  }

  iterator& operator++() {
    container_ = container_->next();
    return *this;
  }

  iterator operator++(int) {
    iterator result(*this);
    ++*this;
    return result;
  }

  iterator& operator--() {
    container_ = container_->prev();
    return *this;
  }

  iterator operator--(int) {
    iterator result(*this);
    --*this;
    return result;
  }

  embedded_list_iterator<LinkField, false> unconst() const {
    return embedded_list_iterator<LinkField, false>(container_);
  }

  reference operator*() const { return *operator->(); }

  pointer operator->() const { return LinkField::StructPtr4FieldPtr(container_); }

  bool operator==(const iterator& other) { return this->container_ == other.container_; }

  friend bool operator==(const iterator& l, const iterator& r) {
    return l.container_ == r.container_;
  }

  bool operator!=(const iterator& other) { return !(*this == other); }

  friend bool operator!=(const iterator& l, const iterator& r) { return !(l == r); }

 private:
  field_type* container_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_OBJECT_MSG_EMBEDDED_LIST_ITERATOR_H_
