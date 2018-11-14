#include "oneflow/core/kernel/broadcast_add_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastAddConf, BroadcastAddKernel,
                           ARITHMETIC_DATA_TYPE_SEQ);
}
