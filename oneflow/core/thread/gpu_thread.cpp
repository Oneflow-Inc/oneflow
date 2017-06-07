#include "oneflow/core/thread/gpu_thread.h"
#include "cuda_runtime.h"

namespace oneflow {

GpuThread::GpuThread(int device_phy_id) {
  mut_thread() = std::thread([this, device_phy_id]() {
    cudaSetDevice(device_phy_id);
    cudaStream_t copy_hd_cuda_stream;
    cudaStream_t compute_cuda_stream;
    CHECK_EQ(cudaStreamCreate(&copy_hd_cuda_stream), 0);
    CHECK_EQ(cudaStreamCreate(&compute_cuda_stream), 0);
    ThreadContext ctx;
    ctx.copy_hd_cuda_stream = &copy_hd_cuda_stream;
    ctx.compute_cuda_stream = &compute_cuda_stream;
    PollMsgChannel(ctx);
    CHECK_EQ(cudaStreamDestroy(copy_hd_cuda_stream), 0);
    CHECK_EQ(cudaStreamDestroy(compute_cuda_stream), 0);
  });
}

} // namespace oneflow
