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
#ifndef ONEFLOW_CORE_COMMON_ADD_AND_READ_VECTOR_H_
#define ONEFLOW_CORE_COMMON_ADD_AND_READ_VECTOR_H_

#include <vector>
#include <array>
#include <mutex>
#include <cmath>
#include <glog/logging.h>

namespace oneflow {

// `at` is lock free
template<typename T, int N = 20>
class AddAndReadVector {
 public:
  AddAndReadVector() : size_(0) {}
  ~AddAndReadVector() = default;

  using value_type = T;
  using size_type = size_t;

  // not thread safe.
  size_t size() const { return size_; }

  // lock free.
  const T& at(size_t index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, size_);
    int gran = GetGranularity(index + 1);
    return granularity2vector_[gran].data()[index];
  }

  // lock free.
  T& at(size_t index) {
    CHECK_GE(index, 0);
    CHECK_LT(index, size_);
    int gran = GetGranularity(index + 1);
    return granularity2vector_[gran].data()[index];
  }

  void push_back(const T& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    int granularity = GetGranularity(size_);
    if (granularity2vector_[granularity].size() == (1 << granularity)) {
      int next_granularity = granularity + 1;
      CHECK_LT(next_granularity, N);
      CHECK_EQ(next_granularity, GetGranularity(size_ + 1));
      granularity2vector_[next_granularity].reserve(1 << next_granularity);
      granularity2vector_[next_granularity] = granularity2vector_[granularity];
      granularity = next_granularity;
    } else if (granularity2vector_[granularity].size() > (1 << granularity)) {
      LOG(FATAL) << "fatal bug in AddAndReadVector::push_back";
    } else {
      // do nothing.
    }
    auto* vec = &granularity2vector_[granularity];
    vec->push_back(elem);
    size_ = vec->size();
  }

 private:
  static int GetGranularity(size_t size) {
    if (size == 0) { return 0; }
    return std::ceil(std::log2(size));
  }

  std::atomic<size_t> size_;
  std::mutex mutex_;
  std::array<std::vector<T>, N> granularity2vector_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ADD_AND_READ_VECTOR_H_
