#include "oneflow/core/persistence/persistence_thread_pool.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

void PersistenceThreadPool::Schedule(std::function<void()> fn) {
  eigen_thread_pool_->Schedule(fn);
}

PersistenceThreadPool::PersistenceThreadPool(const Plan& plan) {
  int32_t worker_num = JobDesc::Singleton()->PersistenceWorkerNum();
  if (worker_num <= 0) {
    worker_num = std::min(CalcPersistenceTaskNumOnThisMachine(plan), 64);
  }
  eigen_thread_pool_.reset(
      new Eigen::NonBlockingThreadPoolTempl<Eigen::StlThreadEnvironment>(
          worker_num));
}

int32_t PersistenceThreadPool::CalcPersistenceTaskNumOnThisMachine(
    const Plan& plan) {
  int32_t persistence_task_num = 0;
  for (const TaskProto& task_proto : plan.task()) {
    if (task_proto.machine_id() != MachineCtx::Singleton()->this_machine_id()) {
      continue;
    }
    if (task_proto.thrd_id() != IDMgr::Singleton()->PersistenceThrdId()) {
      continue;
    }
    persistence_task_num += 1;
  }
  return persistence_task_num;
}

}  // namespace oneflow
