#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/resource_desc.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/job/machine_context.h"

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
    FOR_RANGE(int64_t, dev_phy_id, 0, Global<ResourceDesc>::Get()->GpuDeviceNum()) {
      threads_.push_back(new GpuThread(thrd_id++, dev_phy_id));
    }
  }
#endif
  FOR_RANGE(int64_t, i, 0, Global<ResourceDesc>::Get()->CpuDeviceNum()) {
    threads_.push_back(new CpuThread(thrd_id++));
  }
  threads_.push_back(new CpuThread(thrd_id++));  // comm_net
  CreatePersistenceThrd(plan, thrd_id);
  compute_thread_pool_.reset(new ThreadPool(Global<ResourceDesc>::Get()->ComputeThreadPoolSize()));
}

void ThreadMgr::CreatePersistenceThrd(const Plan& plan, int64_t thrd_id) {
  const int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();

  int64_t max_thrd_id = 0;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() == this_machine_id) {
      if (max_thrd_id < task.thrd_id()) { max_thrd_id = task.thrd_id(); }
    }
  }

  for (int64_t i = thrd_id; i <= max_thrd_id; i++) { threads_.push_back(new CpuThread(i)); }
}
}  // namespace oneflow
