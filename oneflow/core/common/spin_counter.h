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
#ifndef ONEFLOW_CORE_COMMON_SPIN_COUNTER_H_
#define ONEFLOW_CORE_COMMON_SPIN_COUNTER_H_

#include <atomic>
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class SpinCounter final {
 public:
  SpinCounter() = delete;
  SpinCounter(const SpinCounter&) = delete;
  SpinCounter(SpinCounter&&) = delete;
  ~SpinCounter() = default;

  explicit SpinCounter(int64_t cnt_val) : cnt_val_(cnt_val) {}

  int64_t Decrease() { return --cnt_val_; }
  void Reset(int64_t cnt_val) { cnt_val_ = cnt_val; }
  Maybe<void> WaitUntilCntEqualZero() const;

 private:
  std::atomic<int64_t> cnt_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SPIN_COUNTER_H_
