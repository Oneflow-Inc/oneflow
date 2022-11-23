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
#ifndef ONEFLOW_CORE_COMMON_NOTIFIER_H_
#define ONEFLOW_CORE_COMMON_NOTIFIER_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

enum NotifierStatus { kNotifierStatusSuccess = 0, kNotifierStatusErrorClosed };

class Notifier final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Notifier);
  Notifier() : notified_cnt_(0), is_closed_(false) {}
  ~Notifier() = default;

  NotifierStatus Notify();
  NotifierStatus WaitAndClearNotifiedCnt();
  void Close();

  Maybe<void> TimedWaitAndClearNotifiedCnt(size_t timeout_seconds);
  Maybe<void> TimedWaitAndClearNotifiedCnt(
      const std::function<Maybe<bool>()>& StopWaitingAfterTimeout);

 private:
  size_t notified_cnt_;
  std::mutex mutex_;
  bool is_closed_;
  std::condition_variable cond_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_COMMON_NOTIFIER_H_
