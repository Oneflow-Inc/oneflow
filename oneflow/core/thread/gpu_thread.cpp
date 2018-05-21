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
      ctx.g_cuda_stream.reset(new CudaStreamHandle(&cb_event_chan_));
      ctx.cb_event_chan = &cb_event_chan_;
      PollMsgChannel(ctx);
    }
    if (buf_ptr) { CudaCheck(cudaFree(buf_ptr)); }
  });
  cb_event_poller_ = std::thread([this]() {
    CudaCBEvent cb_event;
    while (cb_event_chan_.Receive(&cb_event) == 0) {
      CudaCheck(cudaEventSynchronize(cb_event.event));
      cb_event.callback();
      CudaCheck(cudaEventDestroy(cb_event.event));
    }
  });
}

GpuThread::~GpuThread() {
  cb_event_chan_.CloseSendEnd();
  cb_event_chan_.CloseReceiveEnd();
  cb_event_poller_.join();
}

#endif

}  // namespace oneflow
