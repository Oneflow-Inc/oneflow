#include "oneflow/user/kernels/activation_ops_util.h"

namespace oneflow {

#define REGISTER_ACTIVATION_CPU_KERNEL(dtype)   \
  REGISTER_ELU_KERNEL(DeviceType::kCPU, dtype); \
  REGISTER_HARDSWISH_KERNEL(DeviceType::kCPU, dtype);

REGISTER_ACTIVATION_CPU_KERNEL(float);
REGISTER_ACTIVATION_CPU_KERNEL(double);

}