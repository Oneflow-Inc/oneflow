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

ThreadMgr::ThreadMgr() {
  const JobDesc* job_desc = JobDesc::Singleton();
  int64_t thrd_id = 0;
  // device
  FOR_RANGE(int64_t, dev_id, 0, job_desc->resource().device_num_per_machine()) {
    if (job_desc->resource().device_type() == kGPU) {
      threads_.push_back(new GpuThread(thrd_id++, dev_id));
    } else {
      threads_.push_back(new CpuThread(thrd_id++));
    }
  }
  // persistence
  FOR_RANGE(int64_t, i, 0, job_desc->PersistenceWorkerNum()) {
    threads_.push_back(new CpuThread(thrd_id++));
  }
  // boxing
  FOR_RANGE(int64_t, i, 0, job_desc->BoxingWorkerNum()) {
    threads_.push_back(new CpuThread(thrd_id++));
  }
  // comm net
  threads_.push_back(new CpuThread(thrd_id++));
}

}  // namespace oneflow
