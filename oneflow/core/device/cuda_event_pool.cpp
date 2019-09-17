#include "oneflow/core/device/cuda_event_pool.h"

namespace oneflow {

#ifdef WITH_CUDA

namespace {

int32_t GetCurrentDevice() {
  int32_t device;
  CudaCheck(cudaGetDevice(&device));
  return device;
}

}  // namespace

CudaEventPool::CudaEventPool() {
  CudaCheck(cudaGetDeviceCount(&dev_cnt_));
  std::vector<std::mutex> mutex_vec(dev_cnt_);
  mutex_vec_.swap(mutex_vec);
  event_queue_vec_.resize(dev_cnt_);
}

CudaEventPool::~CudaEventPool() {
  FOR_RANGE(int32_t, i, 0, dev_cnt_) {
    CudaCurrentDeviceGuard guard(i);
    for (cudaEvent_t event : event_queue_vec_.at(i)) { cudaEventDestroy(event); }
  }
}

cudaEvent_t CudaEventPool::Get() {
  const int32_t device = GetCurrentDevice();
  CHECK_LT(device, dev_cnt_);
  {
    std::lock_guard<std::mutex> lock(mutex_vec_.at(device));
    std::deque<cudaEvent_t>& deque = event_queue_vec_.at(device);
    if (!deque.empty()) {
      cudaEvent_t event = deque.front();
      deque.pop_front();
      return event;
    }
  }
  cudaEvent_t event;
  CudaCheck(cudaEventCreateWithFlags(&event, cudaEventBlockingSync | cudaEventDisableTiming));
  return event;
}

void CudaEventPool::Put(cudaEvent_t event) {
  const int32_t device = GetCurrentDevice();
  CHECK_LT(device, dev_cnt_);
  std::lock_guard<std::mutex> lock(mutex_vec_.at(device));
  event_queue_vec_.at(device).push_back(event);
}

#endif  // WITH_CUDA

}  // namespace oneflow
