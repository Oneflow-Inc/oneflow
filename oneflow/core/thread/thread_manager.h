#ifndef ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
#define ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_

#include "oneflow/core/common/channel.h"
#include "oneflow/core/common/protobuf.h"
#include "oneflow/core/thread/thread.h"
#include "oneflow/core/thread/thread_pool.h"

namespace oneflow {

class Plan;

class ThreadMgr final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ThreadMgr);
  ThreadMgr() = delete;
  ~ThreadMgr();

  Thread* GetThrd(int64_t thrd_id);

  ThreadPool* compute_thread_pool() { return compute_thread_pool_.get(); }

 private:
  friend class Global<ThreadMgr>;
  explicit ThreadMgr(const Plan& plan);

  void CreatePersistenceThrd(const Plan& plan, int64_t thrd_id);

  std::vector<Thread*> threads_;
  std::unique_ptr<ThreadPool> compute_thread_pool_;
};

void SingleThreadLoop(size_t num, std::function<void(size_t i)> Callback);
void MultiThreadLoop(size_t num, std::function<void(size_t i)> Callback);

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_THREAD_MANAGER_H_
