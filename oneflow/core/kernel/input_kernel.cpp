#include "oneflow/core/kernel/input_kernel.h"

namespace oneflow {

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kInputConf, DeviceType::kCPU,
                            InputKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kInputConf, DeviceType::kGPU,
                            InputKernel<DeviceType::kGPU>);
}  // namespace oneflow
