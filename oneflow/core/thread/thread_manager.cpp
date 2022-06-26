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

namespace oneflow {

ThreadMgr::~ThreadMgr() {
  for (auto& thread_pair : threads_) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    thread_pair.second->GetMsgChannelPtr()->Send(msg);
    thread_pair.second.reset();
    VLOG(1) << " Actor thread: " << thread_pair.first << " finished when process exits.";
  }
}

Thread* ThreadMgr::GetThrd(int64_t thrd_id) {
  auto iter = threads_.find(thrd_id);
  CHECK(iter != threads_.end()) << " Thread: " << thrd_id << " not found";
  return iter->second.get();
}

void ThreadMgr::AddThreads(const HashSet<int64_t>& thread_ids) {
  const int64_t this_rank = GlobalProcessCtx::Rank();
  for (int64_t thrd_id : thread_ids) {
    const auto& it = threads_.find(thrd_id);
    if (it != threads_.end()) {
      // NOTE(chengcheng): check thread is not null.
      CHECK(it->second) << " RuntimeError! Thread: " << thrd_id << " in manager must be NOT null.";
      VLOG(1) << " Actor thread: " << thrd_id << " reused.";
      continue;
    }
    StreamId stream_id = DecodeStreamIdFromInt64(thrd_id);
    if (stream_id.rank() != this_rank) { continue; }
    Thread* thread = new Thread(stream_id);
    CHECK_NOTNULL(thread);
    threads_[thrd_id].reset(thread);
    VLOG(1) << " Actor thread: " << thrd_id << " created.";
  }
}

void ThreadMgr::DeleteThreads(const HashSet<int64_t>& thread_ids) {
  std::unique_lock<std::mutex> lock(mutex4del_threads_);
  for (int64_t thrd_id : thread_ids) {
    const auto& it = threads_.find(thrd_id);
    CHECK((it != threads_.end()) && (it->second))
        << " RuntimeError! Actor thread: " << thrd_id << " non-existent but want to delete";
    auto& thread = it->second;
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    thread->GetMsgChannelPtr()->Send(msg);
    thread.reset();
    VLOG(1) << " Actor thread: " << thrd_id << " finished when the graph is destructed.";
    threads_.erase(it);
  }
}

void SingleThreadLoop(size_t num, std::function<void(size_t i)> Callback) {
  FOR_RANGE(size_t, i, 0, num) { Callback(i); }
}

}  // namespace oneflow
