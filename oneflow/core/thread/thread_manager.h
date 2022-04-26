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
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/common/balanced_splitter.h"
#include "oneflow/core/common/cpp_attribute.h"
#include "oneflow/core/thread/thread.h"
#include "oneflow/core/thread/thread_pool.h"
#include "oneflow/core/platform/include/pthread_fork.h"

namespace oneflow {

class Plan;

class ThreadMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadMgr);
  ThreadMgr() = default;
  ~ThreadMgr();

  void AddPlan(const Plan& plan);
  Thread* GetThrd(int64_t thrd_id);

 private:
  friend class Global<ThreadMgr>;

  HashMap<int64_t, std::unique_ptr<Thread>> threads_;
};

void SingleThreadLoop(size_t num, std::function<void(size_t i)> Callback);

template<typename DoEachT>
void MultiThreadLoop(size_t num, const DoEachT& DoEach) {
  if (num == 0) { return; }
  if (unlikely(pthread_fork::IsForkedSubProcess())) {
    SingleThreadLoop(num, DoEach);
    return;
  }
  size_t thread_num = Global<ThreadPool>::Get()->thread_num();
  thread_num = std::min(num, thread_num);
  BalancedSplitter bs(num, thread_num);
  BlockingCounter bc(thread_num);
  FOR_RANGE(size_t, range_id, 0, thread_num) {
    Global<ThreadPool>::Get()->AddWork([&bc, &bs, range_id, DoEach] {
      size_t start = bs.At(range_id).begin();
      size_t end = bs.At(range_id).end();
      FOR_RANGE(size_t, i, start, end) { DoEach(i); }
      bc.Decrease();
    });
  }
  // buzy loop wait.
  bc.WaitForeverUntilCntEqualZero();
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
