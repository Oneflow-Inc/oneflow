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
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

ThreadMgr::~ThreadMgr() {
  for (auto& thread_pair : threads_) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    thread_pair.second->GetMsgChannelPtr()->Send(msg);
    thread_pair.second.reset();
    LOG(INFO) << "actor thread " << thread_pair.first << " finish";
  }
}

Thread* ThreadMgr::GetThrd(int64_t thrd_id) {
  auto iter = threads_.find(thrd_id);
  CHECK(iter != threads_.end()) << "thread " << thrd_id << " not found";
  return iter->second.get();
}

void ThreadMgr::AddPlan(const Plan& plan) {
  const int64_t this_rank = GlobalProcessCtx::Rank();
  for (const TaskProto& task : plan.task()) {
    TaskId task_id = DecodeTaskIdFromInt64(task.task_id());
    StreamId stream_id = task_id.stream_id();
    if (stream_id.rank() != this_rank) { continue; }
    int64_t thrd_id = EncodeStreamIdToInt64(stream_id);
    if (threads_.find(thrd_id) != threads_.end()) { continue; }
    Thread* thread = new Thread(stream_id);
    CHECK_NOTNULL(thread);
    threads_[thrd_id].reset(thread);
  }
}

void SingleThreadLoop(size_t num, std::function<void(size_t i)> Callback) {
  FOR_RANGE(size_t, i, 0, num) { Callback(i); }
}

}  // namespace oneflow
