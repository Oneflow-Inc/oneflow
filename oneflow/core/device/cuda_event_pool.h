#ifndef ONEFLOW_CORE_DEVICE_CUDA_EVENT_POOL_H_
#define ONEFLOW_CORE_DEVICE_CUDA_EVENT_POOL_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/device/cuda_util.h"

namespace oneflow {

#ifdef WITH_CUDA

class CudaEventPool final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudaEventPool);
  ~CudaEventPool();

  cudaEvent_t Get();
  void Put(cudaEvent_t event);

 private:
  friend class Global<CudaEventPool>;
  CudaEventPool();

  std::vector<std::mutex> mutex_vec_;
  std::vector<std::deque<cudaEvent_t>> event_queue_vec_;
  int32_t dev_cnt_;
};

#endif  // WITH_CUDA

}  // namespace oneflow

#endif  // ONEFLOW_CORE_DEVICE_CUDA_EVENT_POOL_H_
