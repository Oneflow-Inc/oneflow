#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_WORKER_POOL_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_WORKER_POOL_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/device/cpu_worker.h"

namespace oneflow {

class PersistenceWorkerPool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistenceWorkerPool);
  PersistenceWorkerPool() = delete;
  ~PersistenceWorkerPool() = default;

  OF_SINGLETON(PersistenceWorkerPool);

  std::tuple<CpuWorker*, int32_t> AllocateOneWorker();

 private:
  PersistenceWorkerPool(const Plan& plan);
  int32_t CalcWorkerNum(const Plan& plan);

  std::vector<CpuWorker> workers_;
  int32_t cur_worker_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_WORKER_POOL_H_
