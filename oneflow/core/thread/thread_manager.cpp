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

Thread* ThreadMgr::GetThrd(int64_t thrd_id) {
  auto it = std::find_if(threads_.begin(), threads_.end(),
                         [thrd_id](const Thread* thrd) { return thrd->ThreadId() == thrd_id; });
  assert(it != threads_.end());

  return *it;
}

ThreadMgr::ThreadMgr(const Plan& plan) {
  const JobDesc* job_desc = Global<JobDesc>::Get();
  int64_t gpu_thrd_id = job_desc->base_id_of_gpu();

  const OneMachineBufInfo& info = plan.buf_info().Get(Global<MachineCtx>::Get()->this_machine_id());

#ifdef WITH_CUDA
  FOR_RANGE(int64_t, i, 0, 4) {
    FOR_RANGE(int64_t, dev_phy_id, 0, job_desc->GpuDeviceNum()) {
      threads_.push_back(new GpuThread(gpu_thrd_id, dev_phy_id, info.buf_size(gpu_thrd_id)));
      gpu_thrd_id += 1;
    }
  }
#endif
  // cpu compute thread
  int64_t compute_thrd_id = job_desc->base_id_of_cpu_compute();
  FOR_RANGE(int64_t, i, 0, job_desc->CpuDeviceNum()) {
    threads_.push_back(new CpuThread(compute_thrd_id, info.buf_size(gpu_thrd_id + i)));
    compute_thrd_id += 1;
  }

  // record load thread and lossprint thread
  int64_t record_load_thrd_id = job_desc->base_id_of_record_load();
  int64_t lossprint_thrd_id = job_desc->base_id_of_loss_print();
  for (const TaskProto& task : plan.task()) {
    if (task.task_type() == TaskType::kRecordLoad) {
      threads_.push_back(new CpuThread(record_load_thrd_id++, 0));
    } else if (task.task_type() == TaskType::kLossPrint) {
      threads_.push_back(new CpuThread(lossprint_thrd_id++, 0));
    }
  }

  // mdsave thread
  int64_t mdsave_thrd_id = job_desc->base_id_of_mdsave();
  FOR_RANGE(int64_t, i, 0, job_desc->PersistenceWorkerNum()) {
    threads_.push_back(new CpuThread(mdsave_thrd_id++, 0));
  }

  // comm_net
  int64_t comm_net_thrd_id = job_desc->base_id_of_comm_net();
  threads_.push_back(new CpuThread(comm_net_thrd_id++, 0));

  // thread pool
  compute_thread_pool_.reset(new ThreadPool(job_desc->CpuDeviceNum()));
}

}  // namespace oneflow
