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
#include "oneflow/core/common/notifier.h"
#include "oneflow/core/common/foreign_lock_helper.h"
#include "oneflow/core/common/env_var/env_var.h"

namespace oneflow {

NotifierStatus Notifier::Notify() {
  bool notify = false;
  {
    std::unique_lock<std::mutex> lock(mutex_);
    if (is_closed_) { return kNotifierStatusErrorClosed; }
    notify = (notified_cnt_ == 0);
    ++notified_cnt_;
  }
  if (notify) { cond_.notify_one(); }
  return kNotifierStatusSuccess;
}

NotifierStatus Notifier::WaitAndClearNotifiedCnt() {
  std::unique_lock<std::mutex> lock(mutex_);
  cond_.wait(lock, [this]() { return notified_cnt_ > 0 || is_closed_; });
  if (notified_cnt_ == 0) { return kNotifierStatusErrorClosed; }
  notified_cnt_ = 0;
  return kNotifierStatusSuccess;
}

Maybe<void> Notifier::TimedWaitAndClearNotifiedCnt(size_t timeout_seconds) {
  return Singleton<ForeignLockHelper>::Get()->WithScopedRelease([&, this]() -> Maybe<void> {
    std::chrono::duration<size_t> seconds(timeout_seconds);
    std::unique_lock<std::mutex> lock(mutex_);
    CHECK_OR_RETURN(cond_.wait_for(lock, seconds, [this]() {
      return notified_cnt_ > 0 || is_closed_;
    })) << Error::TimeoutError();
    CHECK_GT_OR_RETURN(notified_cnt_, 0) << "notifier closed.";
    notified_cnt_ = 0;
    return Maybe<void>::Ok();
  });
}

Maybe<void> Notifier::TimedWaitAndClearNotifiedCnt(
    const std::function<Maybe<bool>()>& StopWaitingAfterTimeout) {
  while (true) {
    auto status = TRY(TimedWaitAndClearNotifiedCnt(EnvInteger<ONEFLOW_TIMEOUT_SECONDS>()));
    if (status.IsOk()) { return status; }
    if (!status.error()->has_timeout_error()) { return status; }
    if (JUST(StopWaitingAfterTimeout())) { return status; }
  }
  UNIMPLEMENTED_THEN_RETURN();
}

void Notifier::Close() {
  std::unique_lock<std::mutex> lock(mutex_);
  is_closed_ = true;
  cond_.notify_all();
}

}  // namespace oneflow
