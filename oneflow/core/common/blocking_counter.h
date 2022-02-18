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
#ifndef ONEFLOW_CORE_COMMON_BLOCKING_COUNTER_H_
#define ONEFLOW_CORE_COMMON_BLOCKING_COUNTER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/common/maybe.h"

namespace oneflow {

class BlockingCounter final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(BlockingCounter);
  BlockingCounter() = delete;
  ~BlockingCounter() = default;

  BlockingCounter(int64_t cnt_val) { cnt_val_ = cnt_val; }

  int64_t Increase();
  int64_t Decrease();
  void WaitForeverUntilCntEqualZero();
  Maybe<void> WaitUntilCntEqualZero(size_t timeout_seconds);
  Maybe<void> WaitUntilCntEqualZero(const std::function<Maybe<bool>()>& StopWaitingAfterTimeout);

 private:
  std::mutex mtx_;
  std::condition_variable cond_;
  int64_t cnt_val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BLOCKING_COUNTER_H_
