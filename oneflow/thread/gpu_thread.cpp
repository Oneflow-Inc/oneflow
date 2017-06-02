#include "thread/gpu_thread.h"
#include "cuda_runtime.h"

namespace oneflow {

GpuThread::GpuThread(int device_phy_id) {
  mut_thread() = std::thread([this, device_phy_id]() {
    cudaSetDevice(device_phy_id);
    PollMsgChannel();
  });
}

} // namespace oneflow
