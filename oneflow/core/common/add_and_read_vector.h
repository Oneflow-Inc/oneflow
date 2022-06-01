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
  AddAndReadVector() : granularity_(0) {}
  ~AddAndReadVector() = default;

  using value_type = T;
  using size_type = size_t;

  // not thread safe.
  size_t size() const { return granularity2vector_[granularity_].size(); }

  // lock free.
  const T& at(size_t index) const {
    CHECK_GE(index, 0);
    int gran = std::log2(index * 2 + 1);
    CHECK_LE(gran, granularity_);
    return granularity2vector_[gran].at(index);
  }

  // lock free.
  T& at(size_t index) {
    CHECK_GE(index, 0);
    int gran = std::log2(index * 2 + 1);
    CHECK_LE(gran, granularity_);
    return granularity2vector_[gran].at(index);
  }

  void push_back(const T& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    if (granularity2vector_[granularity_].size() == (1 << granularity_)) {
      int next_granularity = granularity_ + 1;
      CHECK_LT(next_granularity, N);
      granularity2vector_[next_granularity].reserve(1 << next_granularity);
      granularity2vector_[next_granularity] = granularity2vector_[granularity_];
      granularity_ = next_granularity;
    } else if (granularity2vector_[granularity_].size() > (1 << granularity_)) {
      LOG(FATAL) << "fatal bug in AddAndReadVector::EmplaceBack";
    } else {
      // do nothing
    }
    granularity2vector_[granularity_].push_back(elem);
  }

 private:
  std::atomic<int> granularity_;
  std::mutex mutex_;
  std::array<std::vector<T>, N> granularity2vector_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ADD_AND_READ_VECTOR_H_
