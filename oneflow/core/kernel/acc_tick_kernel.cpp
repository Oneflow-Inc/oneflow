#include "oneflow/core/kernel/acc_tick_kernel.h"

namespace oneflow {

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kAccTickConf, AccTickKernel);

}  // namespace oneflow
