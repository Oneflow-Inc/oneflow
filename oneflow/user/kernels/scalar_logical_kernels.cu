#include "oneflow/user/kernels/scalar_logical_kernels.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.cuh"

namespace oneflow {

#define REGISTER_SCALAR_LOGICAL_GPU_KERNEL(dtype)           \
  REGISTER_SCALAR_LOGICAL_EQUAL_KERNEL(DeviceType::kGPU, dtype);         \
  REGISTER_SCALAR_LOGICAL_NOTEQUAL_KERNEL(DeviceType::kGPU, dtype);         \
  REGISTER_SCALAR_LOGICAL_GREATER_KERNEL(DeviceType::kGPU, dtype);   \
  REGISTER_SCALAR_LOGICAL_GREATER_EQUAL_KERNEL(DeviceType::kGPU, dtype); \
  REGISTER_SCALAR_LOGICAL_LESS_KERNEL(DeviceType::kGPU, dtype); \
  REGISTER_SCALAR_LOGICAL_LESS_EQUAL_KERNEL(DeviceType::kGPU, dtype); 

REGISTER_SCALAR_LOGICAL_GPU_KERNEL(float);
REGISTER_SCALAR_LOGICAL_GPU_KERNEL(double);

}  // namespace oneflow