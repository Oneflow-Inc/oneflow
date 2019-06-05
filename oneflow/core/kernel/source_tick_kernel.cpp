#include "oneflow/core/kernel/source_tick_kernel.h"

namespace oneflow {

REGISTER_KERNEL(OperatorConf::kSourceTickConf, SourceTickKernel);

}  // namespace oneflow
