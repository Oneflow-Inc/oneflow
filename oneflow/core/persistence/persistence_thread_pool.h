#ifndef ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_THREAD_POOL_H_
#define ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_THREAD_POOL_H_

#include "oneflow/core/job/plan.pb.h"
#include "oneflow/core/common/util.h"

namespace oneflow {

class CpuStream final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuStream);
  CpuStream() = default;
  ~CpuStream() = default;

  void Schedule(std::function<void()> fn) { TODO(); }

 private:
};

class PersistenceThreadPool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(PersistenceThreadPool);
  PersistenceThreadPool() = delete;
  ~PersistenceThreadPool() = default;

  OF_SINGLETON(PersistenceThreadPool);

  CpuStream* NewCpuStream() { TODO(); }

 private:
  PersistenceThreadPool(const Plan& plan);
  int32_t CalcThreadNum(const Plan& plan);
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_PERSISTENCE_PERSISTENCE_THREAD_POOL_H_
