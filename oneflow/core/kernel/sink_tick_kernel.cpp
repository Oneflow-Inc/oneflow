#include "oneflow/core/kernel/sink_tick_kernel.h"

namespace oneflow {

REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kSinkTickConf, SinkTickKernel);

}  // namespace oneflow
