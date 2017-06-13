#ifndef ONEFLOW_CORE_UNIQUE_CUDNN_HANDLE_H_
#define ONEFLOW_CORE_UNIQUE_CUDNN_HANDLE_H_

#include "oneflow/core/common/util.h"

namespace oneflow {

class UniqueCudnnHandle final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(UniqueCudnnHandle);
  UniqueCudnnHandle() = delete;

  UniqueCudnnHandle(const cudaStream_t* cuda_stream) {
    CHECK_EQ(cudnnCreate(&handle_), CUDNN_STATUS_SUCCESS);
    CHECK_EQ(cudnnSetStream(handle_, *cuda_stream), CUDNN_STATUS_SUCCESS);
  }
  
  ~UniqueCudnnHandle() {
    cudnnDestroy(handle_);
  }

  const cudnnHandle_t* get() const { return &handle_; }

 private:
  cudnnHandle_t handle_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_UNIQUE_CUDNN_HANDLE_H_
