#include "oneflow/core/kernel/tick_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kTickConf, TickKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
