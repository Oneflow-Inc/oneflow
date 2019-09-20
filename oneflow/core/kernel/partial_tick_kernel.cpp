#include "oneflow/core/kernel/partial_tick_kernel.h"

namespace oneflow {

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kPartialTickConf, PartialTickKernel);

}  // namespace oneflow
