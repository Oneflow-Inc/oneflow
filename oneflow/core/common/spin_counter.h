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

Maybe<void> SpinWaitUntilTimeout(const std::function<bool()>& NeedSpin, int64_t seconds,
                                 const std::function<void()>& HeartbeatCallback,
                                 int64_t heartbeat_interval_seconds);
Maybe<void> SpinWaitUntilTimeout(const std::function<bool()>& NeedSpin, int64_t seconds);

class SpinCounter final {
 public:
  SpinCounter() = delete;
  SpinCounter(const SpinCounter&) = delete;
  SpinCounter(SpinCounter&&) = delete;
  ~SpinCounter() = default;

  explicit SpinCounter(int64_t cnt_val)
      : cnt_val_(cnt_val), timeout_seconds_(5 * 60), heartbeat_interval_seconds_(3) {}
  SpinCounter(int64_t cnt_val, int64_t timeout_seconds, int64_t heartbeat_interval_seconds)
      : cnt_val_(cnt_val),
        timeout_seconds_(timeout_seconds),
        heartbeat_interval_seconds_(heartbeat_interval_seconds) {}
  static Maybe<void> SpinWait(
      int64_t cnt, const std::function<Maybe<void>(const std::shared_ptr<SpinCounter>&)>& Callback);
  static Maybe<void> SpinWait(
      int64_t cnt, const std::function<Maybe<void>(const std::shared_ptr<SpinCounter>&)>& Callback,
      const std::function<void()>& HeartbeatCallback);

  int64_t TimeoutSeconds() const { return timeout_seconds_; }
  int64_t HearbeatIntervalSeconds() const { return heartbeat_interval_seconds_; }

  int64_t Decrease() { return --cnt_val_; }
  Maybe<void> WaitUntilCntEqualZero() const;
  Maybe<void> WaitUntilCntEqualZero(const std::function<void()>& HeartbeatCallback) const;

 private:
  std::atomic<int64_t> cnt_val_;
  int64_t timeout_seconds_;
  int64_t heartbeat_interval_seconds_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_SPIN_COUNTER_H_
