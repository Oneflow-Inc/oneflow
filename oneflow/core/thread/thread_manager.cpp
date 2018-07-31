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
  const int32_t mdsave_conf_num = Global<JobDesc>::Get()->MdSaveWorkerNum();
  const int64_t this_machine_id = Global<MachineCtx>::Get()->this_machine_id();
  HashMap<std::pair<int64_t, int64_t>, int32_t> machine_task_type2thrd_num;
  std::vector<const TaskProto*> persistence_tasks;
  for (const TaskProto& task : plan.task()) {
    if (task.machine_id() != this_machine_id) { continue; }

    if (task.task_type() == TaskType::kRecordLoad || task.task_type() == TaskType::kLossPrint
        || task.task_type() == TaskType::kMdSave || task.task_type() == TaskType::kPrint
        || task.task_type() == TaskType::kAccuracyPrint) {
      auto key = std::make_pair(this_machine_id, task.task_type());

      if (task.task_type() == TaskType::kMdSave
          && machine_task_type2thrd_num[key] >= mdsave_conf_num)
        continue;

      persistence_tasks.push_back(&task);
      machine_task_type2thrd_num[key]++;
    }
  }

  ThrdIdGenerator generator(machine_task_type2thrd_num);
  for (const TaskProto* task : persistence_tasks) {
    int64_t thrd_id = generator.GenerateThrdId(this_machine_id, task->task_type());
    threads_.push_back(new CpuThread(thrd_id, 0));
  }
}

}  // namespace oneflow
