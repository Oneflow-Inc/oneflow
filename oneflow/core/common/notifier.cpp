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

void Notifier::Close() {
  std::unique_lock<std::mutex> lock(mutex_);
  is_closed_ = true;
  cond_.notify_all();
}

}  // namespace oneflow
