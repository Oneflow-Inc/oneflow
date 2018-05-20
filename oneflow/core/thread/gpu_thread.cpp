#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

#ifdef WITH_CUDA

GpuThread::GpuThread(int64_t thrd_id, int64_t dev_id, size_t buf_size) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, dev_id, buf_size]() {
    CudaCheck(cudaSetDevice(dev_id));
    void* buf_ptr = nullptr;
    if (buf_size > 0) { CudaCheck(cudaMalloc(&buf_ptr, buf_size)); }
    {
      ThreadCtx ctx;
      ctx.buf_ptr = buf_ptr;
      ctx.buf_size = buf_size;
      ctx.cur_thread = this;
      ctx.g_cuda_stream.reset(new CudaStreamHandle);
      PollMsgChannel(ctx);
    }
    if (buf_ptr) { CudaCheck(cudaFree(buf_ptr)); }
  });
  poller_thread_ = std::thread([this]() {
    while (true) {
      CudaEventCB event_cb;
      int status = cuda_event_cb_channel_.Receive(&event_cb);
      if (status != 0) break;
      CudaCheck(cudaEventSynchronize(event_cb.event));
      event_cb.callback();
      CudaCheck(cudaEventDestroy(event_cb.event));
    }
  });
}

GpuThread::~GpuThread() {
  cuda_event_cb_channel_.CloseSendEnd();
  cuda_event_cb_channel_.CloseReceiveEnd();
  poller_thread_.join();
}

#endif

}  // namespace oneflow
