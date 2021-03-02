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
#ifndef ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
#define ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/common/auto_registration_factory.h"
#include "oneflow/core/thread/thread.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

class Plan;

class ThreadMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadMgr);
  ThreadMgr() = delete;
  ~ThreadMgr();

  Thread* GetThrd(int64_t thrd_id);

 private:
  friend class Global<ThreadMgr>;
  explicit ThreadMgr(const Plan& plan);

  HashMap<int64_t, std::unique_ptr<Thread>> threads_;
};

void SingleThreadLoop(size_t num, std::function<void(size_t i)> Callback);
void MultiThreadLoop(size_t num, std::function<void(size_t i)> Callback);

#define REGISTER_DEVICE_THREAD_CREATOR_WITH_STREAM_ID(device, creator) \
  REGISTER_CLASS_CREATOR(int, device, Thread, creator, const StreamId&)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
