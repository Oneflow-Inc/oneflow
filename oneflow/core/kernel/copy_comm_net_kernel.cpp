#include "oneflow/core/kernel/copy_comm_net_kernel.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

REGISTER_KERNEL_WITH_NOTHING(OperatorConf::kCopyCommNetConf, CopyCommNetKernel);

}  // namespace oneflow
