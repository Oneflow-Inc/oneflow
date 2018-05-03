#include "oneflow/core/thread/gpu_thread.h"
#include "oneflow/core/device/cuda_stream_handle.h"

namespace oneflow {

#ifdef WITH_CUDA

GpuThread::GpuThread(int64_t thrd_id, int64_t dev_id) {
  set_thrd_id(thrd_id);
  mut_actor_thread() = std::thread([this, dev_id]() {
    CudaCheck(cudaSetDevice(dev_id));
    CudaStreamHandle copy_h2d_cuda_handle;
    CudaStreamHandle copy_d2h_cuda_handle;
    ThreadCtx ctx;
    ctx.copy_h2d_cuda_stream = copy_h2d_cuda_handle.cuda_stream();
    ctx.copy_d2h_cuda_stream = copy_d2h_cuda_handle.cuda_stream();
    PollMsgChannel(ctx);
  });
}

#endif

}  // namespace oneflow
