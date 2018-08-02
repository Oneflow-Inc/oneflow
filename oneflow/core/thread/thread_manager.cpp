#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/thread/cpu_thread.h"
#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/job/thrd_id_generator.h"

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
  threads_.push_back(new CpuThread(thrd_id++, 0));  // comm_net
  CreatePersistenceThrd(plan);
  compute_thread_pool_.reset(new ThreadPool(job_desc->CpuDeviceNum()));
}

void ThreadMgr::CreatePersistenceThrd(const Plan& plan) {
  const int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  std::vector<int64_t> persistence_thrd_ids;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != this_machine_id) { continue; }

    if (ThrdIdGenerator::IsPesistence(task.task_type())) {
      persistence_thrd_ids.push_back(task.thrd_id());
    }
  }

  auto unique = [](std::vector<int64_t>& vec) {
    std::set<int64_t> temp;

    auto removed_start = std::remove_if(vec.begin(), vec.end(), [&temp](const int64_t& value) {
      if (temp.find(value) != std::end(temp)) return true;

      temp.insert(value);
      return false;
    });

    vec.erase(removed_start, vec.end());

    return vec.size();
  };
  unique(persistence_thrd_ids);

  threads_.resize(threads_.size() + persistence_thrd_ids.size());
  for (int64_t thrd_id : persistence_thrd_ids) { threads_[thrd_id] = new CpuThread(thrd_id, 0); }
}

}  // namespace oneflow
