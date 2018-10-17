#ifndef ONEFLOW_CORE_THREAD_GPU_THREAD_H_
#define ONEFLOW_CORE_THREAD_GPU_THREAD_H_

#include "oneflow/core/thread/thread.h"

namespace oneflow {

#ifdef WITH_CUDA

class GpuThread final : public Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GpuThread);
  GpuThread() = delete;
  ~GpuThread();

  GpuThread(int64_t thrd_id, int64_t dev_id);

 private:
  Channel<CudaCBEvent> cb_event_chan_;
  std::thread cb_event_poller_;
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_GPU_THREAD_H_
