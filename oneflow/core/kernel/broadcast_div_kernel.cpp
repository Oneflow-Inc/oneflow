#include "oneflow/core/kernel/broadcast_div_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kBroadcastDivConf, BroadcastDivKernel,
                           FLOATING_DATA_TYPE_SEQ);
}
