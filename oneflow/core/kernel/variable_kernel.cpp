#include "oneflow/core/kernel/variable_kernel.h"

namespace oneflow {

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kVariableConf, VariableKernel, ARITHMETIC_DATA_TYPE_SEQ);

}  // namespace oneflow
