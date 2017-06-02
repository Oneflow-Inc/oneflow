#ifndef ONEFLOW_THREAD_GPU_THREAD_H_
#define ONEFLOW_THREAD_GPU_THREAD_H_

#include "oneflow/thread/thread.h"

namespace oneflow {

class GpuThread final : public Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GpuThread);
  GpuThread() = delete;
  ~GpuThread() = default;
  
  GpuThread(int device_phy_id);

 private:
};

} // namespace oneflow

#endif // ONEFLOW_THREAD_GPU_THREAD_H_
