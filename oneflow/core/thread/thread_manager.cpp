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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

ThreadMgr::~ThreadMgr() {
  for (size_t i = 0; i < threads_.size(); ++i) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    threads_[i]->GetMsgChannelPtr()->Send(msg);
    delete threads_[i];
    LOG(INFO) << "actor thread " << i << " finish";
  }
}

Thread* ThreadMgr::GetThrd(int64_t thrd_id) { return threads_.at(thrd_id); }

ThreadMgr::ThreadMgr(const Plan& plan) {
  int64_t thrd_id = 0;

#ifdef WITH_CUDA
  FOR_RANGE(int64_t, i, 0, GetCudaWorkTypeSize()) {
    FOR_RANGE(int64_t, dev_phy_id, 0, (Global<ResourceDesc, ForSession>::Get()->GpuDeviceNum())) {
      threads_.push_back(new GpuThread(thrd_id++, dev_phy_id));
    }
  }
#endif
  FOR_RANGE(int64_t, i, 0, (Global<ResourceDesc, ForSession>::Get()->CpuDeviceNum())) {
    threads_.push_back(new CpuThread(thrd_id++));
  }
  threads_.push_back(new CpuThread(thrd_id++));  // comm_net
  CreatePersistenceThrd(plan, thrd_id);
}

void ThreadMgr::CreatePersistenceThrd(const Plan& plan, int64_t thrd_id) {
  const int64_t this_machine_id = GlobalProcessCtx::Rank();

  int64_t max_thrd_id = 0;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() == this_machine_id) {
      if (max_thrd_id < task.thrd_id()) { max_thrd_id = task.thrd_id(); }
    }
  }

  for (int64_t i = thrd_id; i <= max_thrd_id; i++) { threads_.push_back(new CpuThread(i)); }
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
