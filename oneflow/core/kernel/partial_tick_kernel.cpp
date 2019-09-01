#include "oneflow/core/kernel/partial_tick_kernel.h"

namespace oneflow {

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kPartialTickConf, DeviceType::kCPU,
                            PartialTickKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kPartialTickConf, DeviceType::kGPU,
                            PartialTickKernel<DeviceType::kGPU>);

}  // namespace oneflow
