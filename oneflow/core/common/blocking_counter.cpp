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
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/common/singleton.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

int64_t BlockingCounter::Increase() {
  std::unique_lock<std::mutex> lck(mtx_);
  CHECK_GT(cnt_val_, 0);
  cnt_val_ += 1;
  return cnt_val_;
}

int64_t BlockingCounter::Decrease() {
  std::unique_lock<std::mutex> lck(mtx_);
  cnt_val_ -= 1;
  if (cnt_val_ == 0) { cond_.notify_all(); }
  return cnt_val_;
}

Maybe<void> BlockingCounter::WaitUntilCntEqualZero(size_t timeout_seconds) {
  return Singleton<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
    std::chrono::duration<size_t> seconds(timeout_seconds);
    std::unique_lock<std::mutex> lck(mtx_);
    CHECK_OR_RETURN(cond_.wait_for(lck, seconds, [this]() { return cnt_val_ == 0; }))
        << Error::TimeoutError();
    return Maybe<void>::Ok();
  });
}

void BlockingCounter::WaitForeverUntilCntEqualZero() {
  CHECK_JUST(WaitUntilCntEqualZero([]() -> Maybe<bool> { return false; }));
}

Maybe<void> BlockingCounter::WaitUntilCntEqualZero(
    const std::function<Maybe<bool>()>& StopWaitingAfterTimeout) {
  while (true) {
    auto status = TRY(WaitUntilCntEqualZero(EnvInteger<ONEFLOW_TIMEOUT_SECONDS>()));
    if (status.IsOk()) { return status; }
    if (!status.error()->has_timeout_error()) { return status; }
    if (JUST(StopWaitingAfterTimeout())) { return status; }
  }
  UNIMPLEMENTED_THEN_RETURN();
}

}  // namespace oneflow
