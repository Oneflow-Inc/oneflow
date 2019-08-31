#include "oneflow/core/kernel/acc_tick_kernel.h"

namespace oneflow {

REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kAccTickConf, DeviceType::kCPU,
                            AccTickKernel<DeviceType::kCPU>);
REGISTER_KERNEL_WITH_DEVICE(OperatorConf::kAccTickConf, DeviceType::kGPU,
                            AccTickKernel<DeviceType::kGPU>);

}  // namespace oneflow
