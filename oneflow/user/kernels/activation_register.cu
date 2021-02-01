#include "oneflow/user/kernels/elu_kernel.cuh"

namespace oneflow{

REGISTER_ELU_KERNEL(DeviceType::kGPU, half);
REGISTER_ELU_KERNEL(DeviceType::kGPU, float);
REGISTER_ELU_KERNEL(DeviceType::kGPU, double);

} // namespace oneflow 