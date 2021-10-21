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
#ifndef ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DEVICE_CTX_POLLER_H_
#define ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DEVICE_CTX_POLLER_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/maybe.h"
#include "oneflow/core/device/device_context.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

namespace boxing {

namespace collective {

class CollectiveBoxingDeviceCtxCheckpoint final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingDeviceCtxCheckpoint);
  CollectiveBoxingDeviceCtxCheckpoint() = default;
  ~CollectiveBoxingDeviceCtxCheckpoint() = default;

  void SetCallback(std::function<void()> done_callback) {
    CHECK(!done_callback_);
    done_callback_ = std::move(done_callback);
  }
  void SetDone() {
    CHECK(done_callback_);
    done_callback_();
  }

 private:
  std::function<void()> done_callback_;
};

class CollectiveBoxingDeviceCtxPoller final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingDeviceCtxPoller);
  ~CollectiveBoxingDeviceCtxPoller();

  std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> CreateCheckpoint();
  void Enqueue(const std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint>& checkpoint,
               const std::function<void()>&);

 private:
  friend class Global<CollectiveBoxingDeviceCtxPoller>;
  CollectiveBoxingDeviceCtxPoller();

  std::shared_ptr<HashMap<CollectiveBoxingDeviceCtxCheckpoint*, std::list<std::function<void()>>>>
      checkpoint2callbacks_;
  std::shared_ptr<ThreadPool> thread_pool_;
  std::shared_ptr<std::mutex> mutex_;
};

}  // namespace collective

}  // namespace boxing

}  // namespace oneflow

#endif  // ONEFLOW_CORE_JOB_COLLECTIVE_BOXING_DEVICE_CTX_POLLER_H_
