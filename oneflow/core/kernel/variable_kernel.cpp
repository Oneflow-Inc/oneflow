#include "oneflow/core/kernel/variable_kernel.h"

namespace oneflow {

REGISTER_KERNEL_HELPER_GPU_FLOATING(OperatorConf::kVariableConf, VariableKernel);

}  // namespace oneflow
