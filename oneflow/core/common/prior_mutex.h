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
#ifndef ONEFLOW_CORE_COMMON_PRIOR_MUTEX_H_
#define ONEFLOW_CORE_COMMON_PRIOR_MUTEX_H_

#include <mutex>

namespace oneflow {

template<typename T>
class HighPriorUniqueLock;

template<typename T>
class LowPriorUniqueLock;

class PriorMutex final {
 public:
  PriorMutex() : mutex_(), is_hurry_(false) {}
  ~PriorMutex() = default;

  PriorMutex(const PriorMutex&) = delete;
  PriorMutex(PriorMutex&&) = delete;
  PriorMutex& operator=(const PriorMutex&) = delete;
  PriorMutex& operator=(PriorMutex&&) = delete;

 private:
  friend class HighPriorUniqueLock<PriorMutex>;
  friend class LowPriorUniqueLock<PriorMutex>;

  std::mutex* mut_mutex() { return &mutex_; }
  volatile bool* mut_is_hurry() { return &is_hurry_; }

  std::mutex mutex_;
  // is_hurry_ politely notifies other threads to leave the critical section.
  // It's no need to replace with std::atomic<bool>
  volatile bool is_hurry_;
};

template<>
class LowPriorUniqueLock<PriorMutex> final {
 public:
  explicit LowPriorUniqueLock(PriorMutex& prior_mutex) : prior_mutex_(&prior_mutex) {
    prior_mutex_->mut_mutex()->lock();
  }
  ~LowPriorUniqueLock() { prior_mutex_->mut_mutex()->unlock(); }
  LowPriorUniqueLock(const LowPriorUniqueLock&) = delete;
  LowPriorUniqueLock(LowPriorUniqueLock&&) = delete;
  LowPriorUniqueLock& operator=(const LowPriorUniqueLock&) = delete;
  LowPriorUniqueLock& operator=(LowPriorUniqueLock&&) = delete;

  bool TestIsHurryAndClearHurry() {
    bool is_hurry = *prior_mutex_->mut_is_hurry();
    if (is_hurry) { *prior_mutex_->mut_is_hurry() = false; }
    return is_hurry;
  }

 private:
  PriorMutex* prior_mutex_;
  std::unique_lock<std::mutex> lock_;
};

template<>
class HighPriorUniqueLock<PriorMutex> final {
 public:
  explicit HighPriorUniqueLock(PriorMutex& prior_mutex) : prior_mutex_(&prior_mutex) {
    *prior_mutex_->mut_is_hurry() = true;
    while (!prior_mutex_->mut_mutex()->try_lock()) {}
  }
  ~HighPriorUniqueLock() {
    prior_mutex_->mut_mutex()->unlock();
    *prior_mutex_->mut_is_hurry() = false;
  }
  HighPriorUniqueLock(const HighPriorUniqueLock&) = delete;
  HighPriorUniqueLock(HighPriorUniqueLock&&) = delete;
  HighPriorUniqueLock& operator=(const HighPriorUniqueLock&) = delete;
  HighPriorUniqueLock& operator=(HighPriorUniqueLock&&) = delete;

 private:
  PriorMutex* prior_mutex_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_PRIOR_MUTEX_H_
