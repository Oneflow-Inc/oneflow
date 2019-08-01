#include "oneflow/core/kernel/variable_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kVariableConf, VariableKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
