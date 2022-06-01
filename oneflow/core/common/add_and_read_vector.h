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

#include <deque>
#include <atomic>
#include <mutex>

namespace oneflow {

// `at` is lock free
template<typename T, int N = 20>
class AddAndReadVector {
 public:
  AddAndReadVector() {}
  ~AddAndReadVector() = default;

  using value_type = T;
  using size_type = size_t;

  // not thread safe.
  typename std::deque<T>::size_type size() const { return size_; }

  // lock free.
  const T& at(size_t index) const { return data_.at(index); }

  // lock free.
  T& at(size_t index) { return data_.at(index); }

  void push_back(const T& elem) {
    std::unique_lock<std::mutex> lock(mutex_);
    data_.push_back(elem);
    size_ = data_.size();
  }

 private:
  std::atomic<typename std::deque<T>::size_type> size_;
  std::mutex mutex_;
  std::deque<T> data_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_ADD_AND_READ_VECTOR_H_
