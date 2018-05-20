#ifndef ONEFLOW_CORE_THREAD_GPU_THREAD_H_
#define ONEFLOW_CORE_THREAD_GPU_THREAD_H_

#include "oneflow/core/thread/thread.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA

struct CudaEventCB {
  cudaEvent_t event;
  std::function<void()> callback;
};

class GpuThread final : public Thread {
 public:
  OF_DISALLOW_COPY_AND_MOVE(GpuThread);
  GpuThread() = delete;
  virtual ~GpuThread();

  GpuThread(int64_t thrd_id, int64_t dev_id, size_t buf_size);
  void AddEventCb(const CudaEventCB& event_cb) { cuda_event_cb_channel_.Send(event_cb); }

 private:
  Channel<CudaEventCB> cuda_event_cb_channel_;
  std::thread poller_thread_;
};

#endif

}  // namespace oneflow

#endif  // ONEFLOW_CORE_THREAD_GPU_THREAD_H_
