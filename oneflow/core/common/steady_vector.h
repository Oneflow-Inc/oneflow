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
#ifndef ONEFLOW_CORE_COMMON_STEADY_VECTOR_H_
#define ONEFLOW_CORE_COMMON_STEADY_VECTOR_H_

#include <memory>
#include <array>
#include <mutex>
#include <cmath>
#include <glog/logging.h>

namespace oneflow {

template<typename T, int N = 20>
class SteadyVector {
 public:
  SteadyVector() : size_(0) {}
  ~SteadyVector() = default;

  using value_type = const T;
  using size_type = size_t;

  // thread safe.
  size_t size() const { return size_.load(std::memory_order_acquire); }

  // thread safe.
  const T& at(size_t index) const {
    CHECK_GE(index, 0);
    CHECK_LT(index, size_);
    return (*this)[index];
  }

  // thread safe.
  const T& operator[](size_t index) const {
    int gran = 0;
    size_t start = 0;
    GetGranularityAndStart(index, &gran, &start);
    return granularity2data_[gran].get()[index - start];
  }

  // `index` should be <= size()
  void SetOrAdd(size_t index, T value) {
    std::unique_lock<std::mutex> lock(mutex_);
    size_t size = size_.load(std::memory_order_relaxed);
    CHECK_LE(index, size) << "index out of range";
    if (index == size) {
      int granularity = GetGranularity(size);
      if (size + 1 == (1 << granularity)) {
        CHECK_LT(granularity, N);
        granularity2data_[granularity].reset(new T[1 << granularity]);
      }
      *Mutable(index) = std::move(value);
      size_.fetch_add(1, std::memory_order_release);
    } else {
      *Mutable(index) = std::move(value);
    }
  }

  void push_back(const T& elem) { SetOrAdd(size_, elem); }

 private:
  T* Mutable(size_t index) {
    int gran = 0;
    size_t start = 0;
    GetGranularityAndStart(index, &gran, &start);
    return &granularity2data_[gran].get()[index - start];
  }

  static void GetGranularityAndStart(size_t index, int* gran, size_t* start) {
    *gran = GetGranularity(index);
    *start = (1 << *gran) - 1;
  }

#ifdef __GNUC__
#define LOG2(x) ((unsigned)(8 * sizeof(unsigned long long) - __builtin_clzll((x)) - 1))
#else
#define LOG2(x) std::log2(x)
#endif

  static int GetGranularity(size_t index) { return LOG2(index + 1); }

#undef LOG2

  std::atomic<size_t> size_;
  std::mutex mutex_;
  std::array<std::unique_ptr<T[]>, N> granularity2data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_STEADY_VECTOR_H_
