#include "oneflow/core/kernel/nccl_kernel.h"

namespace oneflow {

REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kNcclReduceScatterConf, NcclReduceScatterKernel);
REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kNcclAllGatherConf, NcclAllGatherKernel);
REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kNcclAllReduceConf, NcclAllReduceKernel);

}  // namespace oneflow
