#include "oneflow/core/kernel/sink_tick_kernel.h"

namespace oneflow {

REGISTER_KERNEL(OperatorConf::kSinkTickConf, SinkTickKernel);

}  // namespace oneflow
