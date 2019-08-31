#include "oneflow/core/kernel/source_tick_kernel.h"

namespace oneflow {

REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kSourceTickConf, SourceTickKernel);

}  // namespace oneflow
