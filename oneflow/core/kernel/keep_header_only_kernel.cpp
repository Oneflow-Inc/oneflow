#include "oneflow/core/kernel/keep_header_only_kernel.h"

namespace oneflow {

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kKeepHeaderOnlyConf, KeepHeaderOnlyKernel);

}  // namespace oneflow
