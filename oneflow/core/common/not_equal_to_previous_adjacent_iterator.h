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
#ifndef ONEFLOW_CORE_COMMON_NOT_EQUAL_TO_PREVIOUS_ADJACENT_ITERATOR_H_
#define ONEFLOW_CORE_COMMON_NOT_EQUAL_TO_PREVIOUS_ADJACENT_ITERATOR_H_

#include <iterator>

namespace oneflow {

#define ITER_DEVICE_FUNC __host__ __device__ __forceinline__

template<typename ValueType, typename UnderlyingT, typename OffsetT = ptrdiff_t>
class NotEqualToPreviousAdjacentIterator {
 public:
  typedef NotEqualToPreviousAdjacentIterator self_type;
  typedef OffsetT difference_type;
  typedef ValueType value_type;
  typedef ValueType* pointer;
  typedef ValueType reference;
  typedef std::random_access_iterator_tag iterator_category;

 private:
  const UnderlyingT* underlying;
  OffsetT offset;

 public:
  ITER_DEVICE_FUNC
  NotEqualToPreviousAdjacentIterator(const UnderlyingT* underlying, OffsetT offset)
      : underlying(underlying), offset(offset) {}

  ITER_DEVICE_FUNC self_type operator++(int) {
    self_type ret = *this;
    offset++;
    return ret;
  }

  ITER_DEVICE_FUNC self_type operator++() {
    offset++;
    return *this;
  }

  ITER_DEVICE_FUNC reference operator*() const {
    return offset == 0 ? 0 : (underlying[offset] == underlying[offset - 1] ? 0 : 1);
  }

  template<typename Distance>
  ITER_DEVICE_FUNC self_type operator+(Distance n) const {
    self_type ret(underlying, offset + n);
    return ret;
  }

  template<typename Distance>
  ITER_DEVICE_FUNC self_type& operator+=(Distance n) {
    offset += n;
    return *this;
  }

  template<typename Distance>
  ITER_DEVICE_FUNC self_type operator-(Distance n) const {
    self_type ret(underlying, offset - n);
    return ret;
  }

  template<typename Distance>
  ITER_DEVICE_FUNC self_type& operator-=(Distance n) {
    offset -= n;
    return *this;
  }

  ITER_DEVICE_FUNC difference_type operator-(self_type other) const {
    return offset - other.offset;
  }

  template<typename Distance>
  ITER_DEVICE_FUNC reference operator[](Distance n) const {
    return *(*this + n);
  }

  ITER_DEVICE_FUNC pointer operator->() { return nullptr; }

  ITER_DEVICE_FUNC bool operator==(const self_type& rhs) {
    return (offset == rhs.offset) && ((underlying == rhs.underlying));
  }

  ITER_DEVICE_FUNC bool operator!=(const self_type& rhs) {
    return offset != rhs.offset || underlying != rhs.underlying;
  }

  friend std::ostream& operator<<(std::ostream& os, const self_type& itr) { return os; }
};

#undef ITER_DEVICE_FUNC

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_NOT_EQUAL_TO_PREVIOUS_ADJACENT_ITERATOR_H_
