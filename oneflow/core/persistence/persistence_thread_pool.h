#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_THREAD_POOL_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_THREAD_POOL_H_

#include "unsupported/Eigen/CXX11/ThreadPool"
#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class PersistenceThreadPool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistenceThreadPool);
  PersistenceThreadPool() = delete;
  ~PersistenceThreadPool() = default;

  OF_SINGLETON(PersistenceThreadPool);

  void Schedule(std::function<void()> fn);

 private:
  PersistenceThreadPool(const Plan& plan);
  int32_t CalcPersistenceTaskNumOnThisMachine(const Plan& plan);

  std::unique_ptr<Eigen::ThreadPoolInterface> eigen_thread_pool_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_THREAD_POOL_H_
