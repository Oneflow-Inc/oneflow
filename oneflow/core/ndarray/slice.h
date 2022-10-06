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
#ifndef ONEFLOW_CORE_NDARRAY_SLICE_H_
#define ONEFLOW_CORE_NDARRAY_SLICE_H_

#include "oneflow/core/ndarray/cpu_ndarray.h"

namespace oneflow {

class Slice final {
 public:
  static const int64_t kStart = LLONG_MIN;
  static const int64_t kEnd = LLONG_MAX;

  Slice(const Slice&) = default;
  Slice(int64_t index) : start_(index), end_(index + 1), stride_(1), value_capacity_(0) {}
  Slice(const std::initializer_list<int64_t>& l);
  ~Slice() = default;

  const Slice& Bound(size_t value_capacity);

  ALWAYS_INLINE int64_t Get(int64_t index) const { return start_ + index * stride_; }
  bool IsBounded() const;
  size_t Size() const;
  bool IsContiguous() const;
  bool IsCoveringAll() const;

 private:
  int64_t start_;
  int64_t end_;
  int64_t stride_;
  size_t value_capacity_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_NDARRAY_SLICE_H_
