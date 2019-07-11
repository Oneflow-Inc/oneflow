#ifndef ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_NEW_KERNEL_UTIL_H_

#include "oneflow/core/common/blas.h"
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/common/str_util.h"
#include "oneflow/core/device/cudnn_util.h"
#include "oneflow/core/job/job_desc.h"
#include "oneflow/core/job/resource.pb.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/operator/op_conf.pb.h"
#include "oneflow/core/persistence/snapshot.h"
#include "oneflow/core/register/blob.h"
#include "oneflow/core/common/switch_func.h"
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
