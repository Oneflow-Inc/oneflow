/*
Copyright 2020 The OneFlow Authors. All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
*/
#include "oneflow/user/kernels/multi_reduce_kernels.h"

namespace oneflow {

#define REGISTER_MULTI_REDUCE_SUM_POW_ABS_CPU_KERNEL(dtype)               \
  REGISTER_USER_KERNEL("multi_reduce_sum_pow_abs")                        \
      .SetCreateFn<MultiReduceSumPowAbsKernel<DeviceType::kCPU, dtype>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)     \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_MULTI_REDUCE_XIMUM_ABS_CPU_KERNEL(op_type_name, ximum_enum, dtype)  \
  REGISTER_USER_KERNEL(op_type_name)                                                 \
      .SetCreateFn<MultiReduceXimumAbsKernel<DeviceType::kCPU, dtype, ximum_enum>>() \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

#define REGISTER_MULTI_REDUCE_XIMUM_ABS_CPU_KERNELS(dtype)                                     \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_CPU_KERNEL("multi_reduce_max_abs", Ximum::kMax, dtype)       \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_CPU_KERNEL("multi_reduce_min_abs", Ximum::kMin, dtype)       \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_CPU_KERNEL("local_multi_reduce_max_abs", Ximum::kMax, dtype) \
  REGISTER_MULTI_REDUCE_XIMUM_ABS_CPU_KERNEL("local_multi_reduce_min_abs", Ximum::kMin, dtype)

REGISTER_MULTI_REDUCE_SUM_POW_ABS_CPU_KERNEL(float)
REGISTER_MULTI_REDUCE_SUM_POW_ABS_CPU_KERNEL(double)

REGISTER_MULTI_REDUCE_XIMUM_ABS_CPU_KERNELS(float)
REGISTER_MULTI_REDUCE_XIMUM_ABS_CPU_KERNELS(double)

}  // namespace oneflow
