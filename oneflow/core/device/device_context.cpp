#include "oneflow/core/device/device_context.h"

namespace oneflow {

DeviceCtx::~DeviceCtx() {
  eigen_gpu_device_.reset();
  eigen_cuda_stream_.reset();
}

}  // namespace oneflow
