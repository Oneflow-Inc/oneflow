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
#ifndef ONEFLOW_CORE_COMMON_PERMUTATION_ITERATOR_H_
#define ONEFLOW_CORE_COMMON_PERMUTATION_ITERATOR_H_

#include <iterator>

namespace oneflow {

#define ITER_DEVICE_FUNC __host__ __device__ __forceinline__

template<typename T, typename DataIter, typename IndexIter, typename OffsetT = std::ptrdiff_t>
class PermutationIterator {
 public:
  using iterator_category = std::random_access_iterator_tag;
  using self_type = PermutationIterator;
  using difference_type = OffsetT;
  using value_type = T;
  using pointer = T*;
  using reference = T&;

  ITER_DEVICE_FUNC PermutationIterator(DataIter data_iter, IndexIter index_iter)
      : data_iter_(data_iter), index_iter_(index_iter) {}

  // const methods

  ITER_DEVICE_FUNC bool operator==(const PermutationIterator& rhs) const {
    return index_iter_ == rhs.index_iter_ && data_iter_ == rhs.data_iter_;
  }

  ITER_DEVICE_FUNC bool operator!=(const PermutationIterator& rhs) const { return !(*this == rhs); }

  template<typename Int>
  ITER_DEVICE_FUNC PermutationIterator operator+(Int n) const {
    return PermutationIterator(data_iter_, index_iter_ + n);
  }

  template<typename Int>
  ITER_DEVICE_FUNC PermutationIterator operator-(Int n) const {
    return PermutationIterator(data_iter_, index_iter_ - n);
  }

  ITER_DEVICE_FUNC difference_type operator-(PermutationIterator other) const {
    return index_iter_ - other.index_iter_;
  }

  ITER_DEVICE_FUNC pointer operator->() const { return &data_iter_[*index_iter_]; }

  ITER_DEVICE_FUNC reference operator*() const { return data_iter_[*index_iter_]; }

  template<typename Int>
  ITER_DEVICE_FUNC reference operator[](Int n) const {
    return data_iter_[index_iter_[n]];
  }

  // mutable methods

  ITER_DEVICE_FUNC PermutationIterator operator++(int) {
    PermutationIterator ret = *this;
    index_iter_++;
    return ret;
  }

  ITER_DEVICE_FUNC PermutationIterator operator++() {
    index_iter_++;
    return *this;
  }

  ITER_DEVICE_FUNC PermutationIterator operator--(int) {
    PermutationIterator ret = *this;
    index_iter_--;
    return ret;
  }

  ITER_DEVICE_FUNC PermutationIterator operator--() {
    index_iter_--;
    return *this;
  }

  template<typename Int>
  ITER_DEVICE_FUNC PermutationIterator& operator+=(Int n) {
    index_iter_ += n;
    return *this;
  }

  template<typename Int>
  ITER_DEVICE_FUNC PermutationIterator& operator-=(Int n) {
    index_iter_ -= n;
    return *this;
  }

  ITER_DEVICE_FUNC pointer operator->() { return &data_iter_[*index_iter_]; }

  ITER_DEVICE_FUNC reference operator*() { return data_iter_[*index_iter_]; }

  template<typename Int>
  ITER_DEVICE_FUNC reference operator[](Int n) {
    return data_iter_[index_iter_[n]];
  }

 private:
  DataIter data_iter_;
  IndexIter index_iter_;
};

#undef ITER_DEVICE_FUNC

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_PERMUTATION_ITERATOR_H_
