#ifndef ONEFLOW_CORE_KERNEL_UTIL_ARITHEMETIC_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_ARITHEMETIC_INTERFACE_H_

#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<DeviceType>
class ArithemeticIf;

}  // namespace oneflow

// #include "oneflow/core/kernel/util/cuda_arithemetic_interface.h"
// #include "oneflow/core/kernel/util/host_arithemetic_interface.h"

#endif  // ONEFLOW_CORE_KERNEL_UTIL_ARITHEMETIC_INTERFACE_H_
