#include "oneflow/core/kernel/copy_comm_net_kernel.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

REGISTER_KERNEL(OperatorConf::kCopyCommNetConf, CopyCommNetKernel);

}  // namespace oneflow
