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
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/rpc/include/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/id_serialization.h"

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

ThreadMgr::ThreadMgr(const Plan& plan) {
  const int64_t this_rank = GlobalProcessCtx::Rank();
  for (const TaskProto& task : plan.task()) {
    TaskId task_id = DeserializeTaskIdFromInt64(task.task_id());
    StreamId stream_id = task_id.stream_id();
    if (stream_id.device_id().rank() != this_rank) { continue; }
    int64_t thrd_id = SerializeStreamIdToInt64(stream_id);
    if (threads_.find(thrd_id) != threads_.end()) { continue; }
    Thread* thread =
        NewObj<int, Thread, const StreamId&>(stream_id.device_id().device_type(), stream_id);
    CHECK_NOTNULL(thread);
    threads_[thrd_id].reset(thread);
  }
}

void SingleThreadLoop(size_t num, std::function<void(size_t i)> Callback) {
  FOR_RANGE(size_t, i, 0, num) { Callback(i); }
}

void MultiThreadLoop(size_t num, std::function<void(size_t i)> Callback) {
  size_t thread_num = Global<ThreadPool>::Get()->thread_num();
  thread_num = std::min(num, thread_num);
  BalancedSplitter bs(num, thread_num);
  BlockingCounter bc(thread_num);
  FOR_RANGE(size_t, range_id, 0, thread_num) {
    Global<ThreadPool>::Get()->AddWork([&bc, &bs, range_id, Callback] {
      FOR_RANGE(size_t, i, bs.At(range_id).begin(), bs.At(range_id).end()) { Callback(i); }
      bc.Decrease();
    });
  }
  bc.WaitUntilCntEqualZero();
}

}  // namespace oneflow
