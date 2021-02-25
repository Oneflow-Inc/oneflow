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
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

namespace {

Thread* NewThread(StreamId stream_id) {
  Thread* thread = nullptr;
  switch (stream_id.device_type()) {
#ifdef WITH_CUDA
    case DeviceType::kGPU: {
      thread = new GpuThread(SerializeStreamIdToInt64(stream_id),
                             static_cast<int64_t>(stream_id.device_index()));
      break;
    }
#endif
    case DeviceType::kCPU: {
      thread = new CpuThread(SerializeStreamIdToInt64(stream_id));
      break;
    }
    default: { UNIMPLEMENTED(); }
  }
  return thread;
}

}  // namespace

ThreadMgr::~ThreadMgr() {
  for (const auto& pair : threads_) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    Thread* thread = pair.second;
    thread->GetMsgChannelPtr()->Send(msg);
    delete thread;
    LOG(INFO) << "actor thread " << SerializeStreamIdToInt64(pair.first) << " finish";
  }
}

Thread* ThreadMgr::GetThrd(int64_t thrd_id) {
  StreamId stream_id = DeserializeStreamIdFromInt64(thrd_id);
  CHECK(threads_.find(stream_id) != threads_.end()) << "thread " << thrd_id << " not found";
  return threads_.at(stream_id);
}

ThreadMgr::ThreadMgr(const Plan& plan) {
  const int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  for (const TaskProto& task : plan.task()) {
    TaskId task_id = DeserializeTaskIdFromInt64(task.task_id());
    if (task_id.process_id().node_index() != this_machine_id) { continue; }
    StreamId stream_id = task_id.stream_id();
    if (threads_.find(stream_id) != threads_.end()) { continue; }
    Thread* thread = NewThread(stream_id);
    CHECK_NOTNULL(thread);
    CHECK(threads_.emplace(stream_id, thread).second);
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
