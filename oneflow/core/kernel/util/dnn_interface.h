#ifndef ONEFLOW_CORE_KERNEL_UTIL_DNN_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_DNN_INTERFACE_H_

#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/common/data_type.h"

namespace oneflow {

template<DeviceType>
struct DnnIf;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_DNN_INTERFACE_H_
