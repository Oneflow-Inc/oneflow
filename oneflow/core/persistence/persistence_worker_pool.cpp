#include "oneflow/core/persistence/persistence_worker_pool.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

std::tuple<CpuWorker*, int32_t> PersistenceWorkerPool::AllocateOneWorker() {
  std::tuple<CpuWorker*, int32_t> ret;
  std::get<0>(ret) = &(workers_.at(cur_worker_id_));
  std::get<1>(ret) = cur_worker_id_;
  return ret;
}

PersistenceWorkerPool::PersistenceWorkerPool(const Plan& plan)
    : workers_(CalcWorkerNum(plan)), cur_worker_id_(0) {}

int32_t PersistenceWorkerPool::CalcWorkerNum(const Plan& plan) {
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
  return std::min(persistence_task_num, 59);
}

}  // namespace oneflow
