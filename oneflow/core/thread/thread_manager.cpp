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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/util.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

ThreadMgr::~ThreadMgr() {
  for (const auto& pair : threads_) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(TaskId(), ActorCmd::kStopThread);
    Thread* thread = pair.second;
    thread->GetMsgChannelPtr()->Send(msg);
    delete thread;
    LOG(INFO) << "actor thread " << pair.first << " finish";
  }
}

Thread* ThreadMgr::GetThrd(uint32_t thrd_id) {
  CHECK(threads_.find(thrd_id) != threads_.end());
  return threads_.at(thrd_id);
}

ThreadMgr::ThreadMgr(const Plan& plan) {
#ifdef WITH_CUDA
  FOR_RANGE(uint32_t, stream_index, 0, StreamIndex::Cuda::kMax) {
    FOR_RANGE(uint32_t, device_index, 0,
              (Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum())) {
      StreamId stream_id = IdUtil::GetStreamId(StreamType::kCudaDevice, device_index, stream_index);
      CHECK(threads_.emplace(stream_id, new GpuThread(stream_id)).second);
    }
  }
#endif
  FOR_RANGE(uint32_t, device_index, 0, (Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum())) {
    StreamId stream_id =
        IdUtil::GetStreamId(StreamType::kCPUDevice, device_index, StreamIndex::CPU::kCompute);
    CHECK(threads_.emplace(stream_id, new CpuThread(stream_id)).second);
  }
  StreamId commnet_stream_id = IdUtil::GetStreamId(StreamType::kCommNet, 0, 0);
  CHECK(threads_.emplace(commnet_stream_id, new CpuThread(commnet_stream_id)).second);
  CreateIndependentThrd(plan);
}

void ThreadMgr::CreateIndependentThrd(const Plan& plan) {
  const int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  for (const TaskProto& task : plan.task()) {
    TaskId task_id(task.task_id());
    if (task_id.process_id().node_index() == this_machine_id
        && task_id.stream_id().stream_type() == StreamType::kIndependent) {
      CHECK(threads_.emplace(task_id.stream_id(), new CpuThread(task_id.stream_id())).second);
    }
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
