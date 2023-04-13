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
#ifndef ONEFLOW_CORE_COMMON_SMALL_VECTOR_H_
#define ONEFLOW_CORE_COMMON_SMALL_VECTOR_H_

#include <glog/logging.h>
#include "llvm/ADT/SmallVector.h"
#include "oneflow/core/common/op_args_reserved_size.h"

namespace oneflow {

template<typename T, size_t N = kOpArgsReservedSize>
class small_vector : public llvm::SmallVector<T, N> {
  using Base = llvm::SmallVector<T, N>;

 public:
  constexpr static size_t kInitialSize = N;
  // https://stackoverflow.com/questions/27954940/a-using-statement-compiles-with-g-fails-compilation-with-clang
  using Base::Base;

  typename Base::reference at(typename Base::size_type idx) {
    CHECK_LT(idx, Base::size());
    return (*this)[idx];
  }
  typename Base::const_reference at(typename Base::size_type idx) const {
    CHECK_LT(idx, Base::size());
    return (*this)[idx];
  }
  typename Base::reference operator[](typename Base::size_type idx) { return this->data()[idx]; }
  typename Base::const_reference operator[](typename Base::size_type idx) const {
    return this->data()[idx];
  }
  typename Base::const_iterator cbegin() const {
    return (typename Base::const_iterator)this->BeginX;
  }
  typename Base::const_iterator cend() const {
    return (typename Base::const_iterator)(this->BeginX) + Base::size();
  }
  typename Base::const_iterator cbegin() { return (typename Base::const_iterator)this->BeginX; }
  typename Base::const_iterator cend() {
    return (typename Base::const_iterator)(this->BeginX) + Base::size();
  }
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SMALL_VECTOR_H_
