#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/job/thrd_id_distributor.h"

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
  const JobDesc* job_desc = Global<JobDesc>::Get();
  int64_t thrd_id = 0;

  const OneMachineBufInfo& info = plan.buf_info().Get(Global<MachineCtx>::Get()->this_machine_id());

#ifdef WITH_CUDA
  FOR_RANGE(int64_t, i, 0, 4) {
    FOR_RANGE(int64_t, dev_phy_id, 0, job_desc->GpuDeviceNum()) {
      threads_.push_back(new GpuThread(thrd_id, dev_phy_id, info.buf_size(thrd_id)));
      thrd_id += 1;
    }
  }
#endif
  FOR_RANGE(int64_t, i, 0, job_desc->CpuDeviceNum()) {
    threads_.push_back(new CpuThread(thrd_id, info.buf_size(thrd_id)));
    thrd_id += 1;
  }

  CreatePersistenceThrd();

  int64_t comm_net_thrd_id = ThrdIdDistributor::get().GenerateThrdId(TaskType::kCopyCommNet, 0);
  threads_.push_back(new CpuThread(comm_net_thrd_id, 0));  // comm_net
  compute_thread_pool_.reset(new ThreadPool(job_desc->CpuDeviceNum()));
}

void ThreadMgr::CreatePersistenceThrd() {
  auto persistence_types = ThrdIdDistributor::get().PersistenceThrdTypes();
  for (auto task_type : persistence_types) {
    auto thrd_ids = ThrdIdDistributor::get().GetThrdIds(task_type);
    for (auto thrd_id : thrd_ids) { threads_.push_back(new CpuThread(thrd_id, 0)); }
  }
}

}  // namespace oneflow
