#include "oneflow/core/kernel/tick_kernel.h"

namespace oneflow {

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kTickConf, DeviceType::kCPU,
                            TickKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kTickConf, DeviceType::kGPU,
                            TickKernel<DeviceType::kGPU>);

}  // namespace oneflow
