#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"

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

void ThreadMgr::ForEachTheadId7MemZoneId7BufSize(
    const OneMachineBufInfo& info, const std::function<void(int64_t, int64_t, size_t)>& DoEach) {
  const JobDesc* job_desc = Global<JobDesc>::Get();
  int64_t thrd_id = 0;

#ifdef WITH_CUDA
  FOR_RANGE(int64_t, i, 0, 4) {
    FOR_RANGE(int64_t, dev_phy_id, 0, job_desc->GpuDeviceNum()) {
      DoEach(thrd_id, dev_phy_id, info.buf_size(thrd_id));
      thrd_id += 1;
    }
  }
#endif
  FOR_RANGE(int64_t, i, 0, job_desc->CpuDeviceNum()) {
    DoEach(thrd_id, job_desc->GpuDeviceNum(), info.buf_size(thrd_id));
    thrd_id += 1;
  }
  FOR_RANGE(int64_t, i, 0, job_desc->PersistenceWorkerNum()) {
    DoEach(thrd_id++, job_desc->GpuDeviceNum(), 0);
  }
  DoEach(thrd_id++, job_desc->GpuDeviceNum(), 0);  // comm_net
}

ThreadMgr::ThreadMgr(const Plan& plan) {
  ForEachTheadId7MemZoneId7BufSize(
      plan.buf_info().Get(Global<MachineCtx>::Get()->this_machine_id()),
      [&](int64_t thrd_id, int64_t device_id, size_t buf_size) {
        if (device_id < Global<JobDesc>::Get()->GpuDeviceNum()) {
          threads_.push_back(new GpuThread(thrd_id, device_id, buf_size));
        } else {
          threads_.push_back(new CpuThread(thrd_id, buf_size));
        }
      });
}

}  // namespace oneflow
