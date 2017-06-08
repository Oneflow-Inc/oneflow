#include "oneflow/core/thread/gpu_thread.h"
#include "cuda_runtime.h"
#include "oneflow/core/common/unique_cuda_stream.h"

namespace oneflow {

GpuThread::GpuThread(int device_phy_id) {
  mut_thread() = std::thread([this, device_phy_id]() {
    cudaSetDevice(device_phy_id);
    UniqueCudaStream copy_hd_cuda_stream;
    UniqueCudaStream compute_cuda_stream;
    ThreadContext ctx;
    ctx.copy_hd_cuda_stream = copy_hd_cuda_stream.get();
    ctx.compute_cuda_stream = compute_cuda_stream.get();
    PollMsgChannel(ctx);
  });
}

} // namespace oneflow
