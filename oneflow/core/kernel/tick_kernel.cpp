#include "oneflow/core/kernel/tick_kernel.h"

namespace oneflow {

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kTickConf, TickKernel);

}  // namespace oneflow
