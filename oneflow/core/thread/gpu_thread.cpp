#include "oneflow/core/thread/gpu_thread.h"
#include "cuda_runtime.h"
#include "oneflow/core/common/unique_cuda_stream.h"
#include "oneflow/core/common/unique_cublas_handle.h"

namespace oneflow {

GpuThread::GpuThread(int device_phy_id) {
  mut_actor_thread() = std::thread([this, device_phy_id]() {
    cudaSetDevice(device_phy_id);
    UniqueCudaStream copy_hd_cuda_stream;
    UniqueCudaStream compute_cuda_stream;
    {
      UniqueCublasHandle cublas_handle(compute_cuda_stream.get());
      ThreadContext ctx;
      ctx.copy_hd_cuda_stream = copy_hd_cuda_stream.get();
      ctx.compute_cuda_stream = compute_cuda_stream.get();
      ctx.cublas_handle = cublas_handle.get();
      PollMsgChannel(ctx);
    }
  });
}

} // namespace oneflow
