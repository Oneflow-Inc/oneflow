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
#ifndef ONEFLOW_CORE_COMMON_FIXED_VECTOR_H_
#define ONEFLOW_CORE_COMMON_FIXED_VECTOR_H_

#include <array>
#include <initializer_list>
#include <vector>
#include <glog/logging.h>
#include "oneflow/core/common/maybe.h"

namespace oneflow {

template<typename _InIter>
using RequireInputIter = typename std::enable_if<
    std::is_convertible<typename std::iterator_traits<_InIter>::iterator_category,
                        std::input_iterator_tag>::value>::type;

template<typename T, int kMaxSize>
class fixed_vector final {
 public:
  using value_type = T;
  using size_type = std::size_t;
  using difference_type = std::ptrdiff_t;
  using reference = value_type&;
  using const_reference = const value_type&;
  using pointer = T*;
  using const_pointer = const T*;
  using iterator = T*;
  using const_iterator = const T*;
  using reverse_iterator = std::reverse_iterator<iterator>;
  using const_reverse_iterator = std::reverse_iterator<const_iterator>;

  fixed_vector() : size_(0) {}
  explicit fixed_vector(size_t size) { assign(size, T()); }
  explicit fixed_vector(size_t size, const T& val) { assign(size, val); }
  template<class InputIt, typename = RequireInputIter<InputIt>>
  fixed_vector(InputIt first, InputIt last) {
    assign(first, last);
  }
  fixed_vector(const fixed_vector& rhs) { *this = rhs; }
  fixed_vector(fixed_vector&& rhs) { *this = std::move(rhs); }
  fixed_vector(std::initializer_list<T> rhs) { assign(rhs); }
  ~fixed_vector() = default;

  fixed_vector& operator=(const fixed_vector& rhs) {
    size_ = rhs.size();
    CheckSize();
    std::copy(rhs.begin(), rhs.end(), begin());
    return *this;
  }
  fixed_vector& operator=(fixed_vector&& rhs) noexcept {
    size_ = rhs.size();
    CheckSize();
    std::copy(rhs.begin(), rhs.end(), begin());
    return *this;
  }
  fixed_vector& operator=(std::initializer_list<T> ilist) {
    size_ = ilist.size();
    assign(ilist);
    return *this;
  }
  void assign(size_type count, const value_type& value) {
    size_ = count;
    CheckSize();
    std::fill(begin(), begin() + size_, value);
  }
  template<class InputIt, typename = RequireInputIter<InputIt>>
  void assign(InputIt first, InputIt last) {
    size_ = last - first;
    CheckSize();
    std::copy(first, last, begin());
  }
  void assign(std::initializer_list<T> ilist) {
    size_ = ilist.size();
    CheckSize();
    std::copy(ilist.begin(), ilist.end(), begin());
  }

  reference at(size_type pos) {
    CHECK_JUST(CheckPos(pos));
    return data_.at(pos);
  }
  const_reference at(size_type pos) const {
    CHECK_JUST(CheckPos(pos));
    return data_.at(pos);
  }

  reference operator[](size_type pos) {
    CHECK_JUST(CheckPos(pos));
    return data_[pos];
  }
  const_reference operator[](size_type pos) const {
    CHECK_JUST(CheckPos(pos));
    return data_[pos];
  }

  reference front() {
    CHECK_JUST(CheckPos(0));
    return data_.at(0);
  }
  const_reference front() const {
    CHECK_JUST(CheckPos(0));
    return data_.at(0);
  }

  reference back() {
    CHECK_JUST(CheckPos(0));
    return data_.at(size_ - 1);
  }
  const_reference back() const {
    CHECK_JUST(CheckPos(0));
    return data_.at(size_ - 1);
  }

  T* data() noexcept { return data_.data(); }
  const T* data() const noexcept { return data_.data(); }

  iterator begin() noexcept { return data_.data(); }
  const_iterator begin() const noexcept { return data_.data(); }
  const_iterator cbegin() const noexcept { return data_.data(); }

  iterator end() noexcept { return data_.data() + size_; }
  const_iterator end() const noexcept { return data_.data() + size_; }
  const_iterator cend() const noexcept { return data_.data() + size_; }

  reverse_iterator rbegin() noexcept { return reverse_iterator(end()); }
  const_reverse_iterator rbegin() const noexcept { return const_reverse_iterator(end()); }
  const_reverse_iterator crbegin() const noexcept { return const_reverse_iterator(cend()); }

  reverse_iterator rend() noexcept { return reverse_iterator(begin()); }
  const_reverse_iterator rend() const noexcept { return const_reverse_iterator(begin()); }
  const_reverse_iterator crend() const noexcept { return const_reverse_iterator(cbegin()); }

  bool empty() const noexcept { return size_ == 0; }

  size_type size() const noexcept { return size_; }

  size_type max_size() const noexcept { return kMaxSize; }

  size_type capacity() const noexcept { return kMaxSize; }

  void clear() noexcept { size_ = 0; }

  iterator insert(iterator pos, const T& value) {
    MoveNToEnd(pos, 1);
    *pos = value;
    return pos;
  }
  iterator insert(iterator pos, T&& value) {
    MoveNToEnd(pos, 1);
    *pos = std::move(value);
    return pos;
  }
  iterator insert(iterator pos, size_type count, const T& value) {
    MoveNToEnd(pos, count);
    std::fill(pos, pos + count, value);
    return pos;
  }
  template<class InputIt, typename = RequireInputIter<InputIt>>
  void insert(iterator pos, InputIt first, InputIt last) {
    MoveNToEnd(pos, last - first);
    std::copy(first, last, pos);
  }
  iterator insert(iterator pos, std::initializer_list<T> ilist) {
    MoveNToEnd(pos, ilist.size());
    std::copy(ilist.begin(), ilist.end(), pos);
    return pos;
  }

  template<class... Args>
  iterator emplace(iterator pos, Args&&... args) {
    MoveNToEnd(pos, 1);
    new (&*pos) T(std::forward<Args>(args)...);
    return pos;
  }

  iterator erase(iterator pos) {
    MoveNToBegin(pos + 1, 1);
    return pos;
  }
  iterator erase(iterator first, iterator last) {
    if (first >= last) { return last; }
    MoveNToBegin(last, last - first);
    return first;
  }

  void push_back(const T& value) { insert(end(), value); }
  void push_back(T&& value) { insert(end(), std::move(value)); }

  template<class... Args>
  void emplace_back(Args&&... args) {
    insert(end(), std::forward<Args>(args)...);
  }

  void pop_back() { --size_; }

  void resize(size_type count) { resize(count, T()); }
  void resize(size_type count, const value_type& value) {
    if (count == size_) { return; }
    if (count < size_) {
      erase(begin() + count, end());
      return;
    }
    insert(end(), count - size_, value);
  }

  void swap(fixed_vector& rhs) noexcept {
    fixed_vector tmp;
    tmp = rhs;
    rhs = *this;
    *this = tmp;
  }

  bool operator==(const fixed_vector& rhs) const {
    if (size() != rhs.size()) { return false; }
    return std::equal(begin(), end(), rhs.begin());
  }

  bool operator!=(const fixed_vector& rhs) const { return !(*this == rhs); }

  bool operator>=(const fixed_vector& rhs) const { return !(*this < rhs); }

  bool operator<=(const fixed_vector& rhs) const { return !(*this > rhs); }

  bool operator>(const fixed_vector& rhs) const {
    return std::lexicographical_compare(rhs.begin(), rhs.end(), begin(), end());
  }

  bool operator<(const fixed_vector& rhs) const {
    return std::lexicographical_compare(begin(), end(), rhs.begin(), rhs.end());
  }

 private:
  void CheckSize() const { CHECK_JUST(CheckSize(size_)); }
  Maybe<void> CheckSize(size_t size) const {
    if (size > kMaxSize) { return Error::OutOfRangeError(); }
    return Maybe<void>::Ok();
  }
  Maybe<void> CheckPos(size_t pos) const {
    if (pos >= size_) { return Error::OutOfRangeError(); }
    return Maybe<void>::Ok();
  }
  void MoveNToEnd(iterator first, size_t N) {
    CheckSize(size_ + N);
    iterator old_end = end();
    size_ += N;
    iterator new_end = end();
    std::copy_backward(first, old_end, new_end);
  }
  void MoveNToBegin(iterator last, size_t N) {
    CHECK_JUST(CheckPos(last - N - begin()));
    iterator old_end = end();
    size_ -= N;
    std::copy(last, old_end, last - N);
  }

  size_t size_;
  std::array<T, kMaxSize> data_;
};

template<class T, long kMaxSize>
void swap(fixed_vector<T, kMaxSize>& lhs, fixed_vector<T, kMaxSize>& rhs) {
  return lhs.swap(rhs);
}

#define SHAPE_MAX_AXIS_SIZE 20

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_FIXED_VECTOR_H_
