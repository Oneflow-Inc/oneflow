#include "oneflow/core/kernel/input_kernel.h"

namespace oneflow {

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kInputConf, InputKernel);
}
