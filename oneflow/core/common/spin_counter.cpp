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
#include <chrono>
#include "oneflow/core/common/spin_counter.h"
#include "oneflow/core/common/global.h"
#include "oneflow/core/common/foreign_lock_helper.h"

namespace oneflow {

Maybe<void> SpinCounter::SpinWait(
    int64_t cnt, const std::function<Maybe<void>(const std::shared_ptr<SpinCounter>&)>& Callback,
    const std::function<void()>& HeartbeatCallback) {
  const auto& spin_counter = std::make_shared<SpinCounter>(cnt);
  JUST(Callback(spin_counter));
  JUST(spin_counter->WaitUntilCntEqualZero(HeartbeatCallback));
  return Maybe<void>::Ok();
}

Maybe<void> SpinCounter::SpinWait(
    int64_t cnt, const std::function<Maybe<void>(const std::shared_ptr<SpinCounter>&)>& Callback) {
  return SpinWait(cnt, Callback, [] {});
}

Maybe<void> SpinWaitUntilTimeout(const std::function<bool()>& NeedSpin, int64_t seconds,
                                 const std::function<void()>& HeartbeatCallback,
                                 int64_t heartbeat_interval_seconds) {
  return Global<ForeignLockHelper>::Get()->WithScopedRelease([&]() -> Maybe<void> {
    const auto& start = std::chrono::steady_clock::now();
    auto time_last_heartbeat = std::chrono::steady_clock::now();
    while (NeedSpin()) {
      auto end = std::chrono::steady_clock::now();
      if (std::chrono::duration<double>(end - time_last_heartbeat).count()
          >= heartbeat_interval_seconds) {
        HeartbeatCallback();
        time_last_heartbeat = end;
      }
      std::chrono::duration<double> elapsed_seconds = end - start;
      CHECK_LT_OR_RETURN(elapsed_seconds.count(), seconds)
          << Error::TimeoutError() << "Timeout error at " << seconds << " seconds.";
    }
    return Maybe<void>::Ok();
  });
}

Maybe<void> SpinWaitUntilTimeout(const std::function<bool()>& NeedSpin, int64_t seconds) {
  return SpinWaitUntilTimeout(
      NeedSpin, seconds, [] {}, seconds);
}

Maybe<void> SpinCounter::WaitUntilCntEqualZero(
    const std::function<void()>& HeartbeatCallback) const {
  return Global<ForeignLockHelper>::Get()->WithScopedRelease(
      [this, HeartbeatCallback]() -> Maybe<void> {
        return SpinWaitUntilTimeout([&] { return cnt_val_ > 0; }, TimeoutSeconds(),
                                    HeartbeatCallback, HearbeatIntervalSeconds());
      });
}

Maybe<void> SpinCounter::WaitUntilCntEqualZero() const {
  return WaitUntilCntEqualZero([] {});
}

}  // namespace oneflow
