#ifndef ONEFLOW_CORE_KERNEL_POOLING_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_POOLING_KERNEL_UTIL_H_

#include "oneflow/core/device/cudnn_util.h"

namespace oneflow {

class CudnnPoolingDesc final {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CudnnPoolingDesc);
  CudnnPoolingDesc() = delete;
  ~CudnnPoolingDesc();

  CudnnPoolingDesc(cudnnPoolingMode_t pooling_mode, int dims, const int* window, const int* padding,
                   const int* stride);

  const cudnnPoolingDescriptor_t& Get() const { return val_; }

 private:
  cudnnPoolingDescriptor_t val_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_POOLING_KERNEL_UTIL_H_
