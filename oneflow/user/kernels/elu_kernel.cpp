#include "oneflow/user/kernels/elu_kernel.h"

namespace oneflow {

REGISTER_ELU_KERNEL(DeviceType::kCPU, float);
REGISTER_ELU_KERNEL(DeviceType::kCPU, double);

}