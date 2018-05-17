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
      ctx.compute_cuda_stream.reset(new CudaStreamHandle);
      ctx.copy_h2d_cuda_stream.reset(new CudaStreamHandle);
      ctx.copy_d2h_cuda_stream.reset(new CudaStreamHandle);
      PollMsgChannel(ctx);
    }
    if (buf_ptr) { CudaCheck(cudaFree(buf_ptr)); }
  });
}

#endif

}  // namespace oneflow
