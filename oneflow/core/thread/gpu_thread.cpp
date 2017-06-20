#include "oneflow/core/thread/gpu_thread.h"
#include "cuda_runtime.h"
#include "oneflow/core/common/cuda_stream_handle.h"

namespace oneflow {

GpuThread::GpuThread(int device_phy_id) {
  mut_actor_thread() = std::thread([this, device_phy_id]() {
    cudaSetDevice(device_phy_id);
    CudaStreamHandle copy_hd_cuda_handle;
    ThreadCtx ctx;
    ctx.copy_hd_cuda_stream = copy_hd_cuda_handle.cuda_stream();
    PollMsgChannel(ctx);
  });
}

} // namespace oneflow
