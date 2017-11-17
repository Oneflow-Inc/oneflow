#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"

namespace oneflow {

ThreadMgr::~ThreadMgr() {
  for (size_t i = 0; i < threads_.size(); ++i) {
    ActorMsg msg = ActorMsg::BuildCommandMsg(-1, ActorCmd::kStopThread);
    threads_[i]->GetMsgChannelPtr()->Send(msg);
    threads_[i].reset();
    LOG(INFO) << "actor thread " << i << " finish";
  }
}

Thread* ThreadMgr::GetThrd(int64_t thrd_loc_id) {
  return threads_.at(thrd_loc_id).get();
}

ThreadMgr::ThreadMgr() {
  LOG(INFO) << "ThreadMgr Init";
  // device thread - device_num_per_machine
  int64_t dev_num_per_machine =
      JobDesc::Singleton()->resource().device_num_per_machine();
  int64_t device_type = JobDesc::Singleton()->resource().device_type();
  threads_.reserve(dev_num_per_machine + 3);
  int64_t thrd_loc_id = 0;
  for (int64_t dev_phy_id = 0; dev_phy_id < dev_num_per_machine; ++dev_phy_id) {
    if (device_type == kGPU) {
      threads_.push_back(of_make_unique<GpuThread>(thrd_loc_id++, dev_phy_id));
    } else {
      threads_.push_back(of_make_unique<CpuThread>(thrd_loc_id++));
    }
  }
  // cpu thread - for persistence
  threads_.push_back(of_make_unique<CpuThread>(thrd_loc_id++));
  // cpu thread - for boxing
  threads_.push_back(of_make_unique<CpuThread>(thrd_loc_id++));
  // cpu thread - for commnet
  threads_.push_back(of_make_unique<CpuThread>(thrd_loc_id++));
}

}  // namespace oneflow
