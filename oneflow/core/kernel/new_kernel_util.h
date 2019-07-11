#ifndef ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_

#include "oneflow/core/kernel/util/dnn_interface.h"
#include "oneflow/core/kernel/util/blas_interface.h"
#include "oneflow/core/kernel/util/arithemetic_interface.h"

namespace oneflow {

template<DeviceType deivce_type>
struct NewKernelUtil : public DnnIf<deivce_type>,
                       public BlasIf<deivce_type>,
                       public ArithemeticIf<deivce_type> {};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
