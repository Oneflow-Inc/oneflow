#include "oneflow/core/kernel/device_tick_kernel.h"

namespace oneflow {

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kDeviceTickConf, DeviceTickKernel);

}  // namespace oneflow
