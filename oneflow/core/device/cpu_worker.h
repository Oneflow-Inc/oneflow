#ifndef ONEFLOW_CORE_DEVICE_CPU_WORKER_H_
#define ONEFLOW_CORE_DEVICE_CPU_WORKER_H_

#include "oneflow/core/common/channel.h"

namespace oneflow {

class CpuWorker final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CpuWorker);
  CpuWorker();
  ~CpuWorker();

  void PushWork(std::function<void()> work) { chan_.Send(work); }

 private:
  std::thread thread_;
  Channel<std::function<void()>> chan_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CPU_WORKER_H_
