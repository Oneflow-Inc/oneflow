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
#ifndef ONEFLOW_CORE_COMMON_BLOCKING_THEN_BUSY_H_
#define ONEFLOW_CORE_COMMON_BLOCKING_THEN_BUSY_H_

#include "oneflow/core/common/maybe.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/spin_counter.h"

namespace oneflow {

class BlockingThenBusy final {
 public:
  BlockingThenBusy(const BlockingThenBusy&) = delete;
  BlockingThenBusy(BlockingThenBusy&&) = delete;
  BlockingThenBusy() = delete;
  explicit BlockingThenBusy(int cnt) : blocking_counter_(cnt), spin_counter_(cnt) {}

  BlockingCounter* mut_blocking_counter() { return &blocking_counter_; }
  SpinCounter* mut_spin_counter() { return &spin_counter_; }

  Maybe<void> WaitUntilCntEqualZero(const std::function<Maybe<bool>()>& StopAfterTimeout) {
    JUST(blocking_counter_.WaitUntilCntEqualZero(StopAfterTimeout));
    JUST(spin_counter_.WaitUntilCntEqualZero());
    return Maybe<void>::Ok();
  }

 private:
  BlockingCounter blocking_counter_;
  SpinCounter spin_counter_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_BLOCKING_THEN_BUSY_H_
