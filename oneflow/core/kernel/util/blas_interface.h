#ifndef ONEFLOW_CORE_KERNEL_UTIL_BLAS_INTERFACE_H_
#define ONEFLOW_CORE_KERNEL_UTIL_BLAS_INTERFACE_H_

#include "oneflow/core/job/resource.pb.h"

namespace oneflow {

template<DeviceType>
class BlasIf;

}  // namespace oneflow

// #include "oneflow/core/kernel/util/cuda_blas_interface.h"
// #include "oneflow/core/kernel/util/host_blas_interface.h"

#endif  // ONEFLOW_CORE_KERNEL_UTIL_BLAS_INTERFACE_H_
