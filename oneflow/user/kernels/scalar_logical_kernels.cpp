#include "oneflow/user/kernels/scalar_logical_kernels.h"

namespace oneflow {

#define REGISTER_SCALAR_LOGICAL_CPU_KERNEL(dtype)           \
  REGISTER_SCALAR_LOGICAL_EQUAL_KERNEL(DeviceType::kCPU, dtype);         \
  REGISTER_SCALAR_LOGICAL_NOTEQUAL_KERNEL(DeviceType::kCPU, dtype);         \
  REGISTER_SCALAR_LOGICAL_GREATER_KERNEL(DeviceType::kCPU, dtype);   \
  REGISTER_SCALAR_LOGICAL_GREATER_EQUAL_KERNEL(DeviceType::kCPU, dtype); \
  REGISTER_SCALAR_LOGICAL_LESS_KERNEL(DeviceType::kCPU, dtype); \
  REGISTER_SCALAR_LOGICAL_LESS_EQUAL_KERNEL(DeviceType::kCPU, dtype); 

REGISTER_SCALAR_LOGICAL_CPU_KERNEL(float);
REGISTER_SCALAR_LOGICAL_CPU_KERNEL(double);

}  // namespace oneflow