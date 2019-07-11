#ifndef ONEFLOW_CORE_KERNEL_UTIL_INTERFACE_BASE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_INTERFACE_BASE_H_

#include "oneflow/core/job/resource.pb.h"
namespace oneflow {

template<DeviceType>
struct DnnIf;

template<DeviceType>
struct BlasIf;

template<DeviceType>
class ArithemeticIf;

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_UTIL_INTERFACE_BASE_H_
