#include "oneflow/core/kernel/broadcast_mul_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastMulConf, BroadcastMulKernel,
                           FLOATING_DATA_TYPE_SEQ);
}
