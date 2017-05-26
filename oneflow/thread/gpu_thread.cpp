#include "thread/gpu_thread.h"

namespace oneflow {

GpuThread::GpuThread(int device_phy_id) {
  mut_thread() = std::thread([this, device_phy_id]() {
    cudaSetDevice(device_phy_id);
    cudaStreamCreate(&mut_device_ctx().cuda_stream);
    PollMsgChannel();
    cudaStreamSynchronize(mut_device_ctx().cuda_stream);
    cudaStreamDestroy(mut_device_ctx().cuda_stream);
  });
}

} // namespace oneflow
