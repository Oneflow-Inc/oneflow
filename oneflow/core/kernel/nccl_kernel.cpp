#include "oneflow/core/kernel/nccl_kernel.h"

namespace oneflow {

REGISTER_KERNEL(OperatorConf::kNcclReduceScatterConf, NcclReduceScatterKernel);
REGISTER_KERNEL(OperatorConf::kNcclAllGatherConf, NcclAllGatherKernel);
REGISTER_KERNEL(OperatorConf::kNcclAllReduceConf, NcclAllReduceKernel);

}  // namespace oneflow
