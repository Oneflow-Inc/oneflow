#ifndef ONEFLOW_CORE_THREAD_CPU_THREAD_H_
#define ONEFLOW_CORE_THREAD_CPU_THREAD_H_

#include "oneflow/core/thread/thread.h"

namespace oneflow {

class CpuThread final : public Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuThread);
  CpuThread() = delete;
  ~CpuThread() = default;

  CpuThread(int64_t thrd_id);

 private:
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_CPU_THREAD_H_
