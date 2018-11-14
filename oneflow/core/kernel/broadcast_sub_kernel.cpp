#include "oneflow/core/kernel/broadcast_sub_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastSubConf, BroadcastSubKernel,
                           FLOATING_DATA_TYPE_SEQ);
}
