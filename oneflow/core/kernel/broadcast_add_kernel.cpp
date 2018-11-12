#include "oneflow/core/kernel/broadcast_add_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastAddConf, BroadcastAddKernel,
                           FLOATING_DATA_TYPE_SEQ);
}
