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
#ifndef ONEFLOW_UTIL_OPS_KERNELS_H_
#define ONEFLOW_UTIL_OPS_KERNELS_H_

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/user_op_registry_manager.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/user/kernels/elementwise_xpu_kernel.h"

namespace oneflow {
namespace user_op {
#define UTIL_OPS_DATA_TYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ       \
  INT_DATA_TYPE_SEQ            \
  UNSIGNED_INT_DATA_TYPE_SEQ

template<DeviceType device_type, typename T, typename Enable = void>
struct IsNanFunctor {
  OF_DEVICE_FUNC bool operator()(const T x) const;
};

template<DeviceType device_type, typename T, typename Enable = void>
struct IsInfFunctor {
  OF_DEVICE_FUNC bool operator()(const T x) const;
};

// Only for util ops register. Output name is "out", input name is "in". Output dtype is bool.
#define REGISTER_UTIL_OPS_KERNELS(device, kernel_name, dtype, functor)                          \
  REGISTER_USER_KERNEL(kernel_name)                                                             \
      .SetCreateFn([]() {                                                                       \
        return user_op::NewOpKernel<                                                            \
            UnaryElemwiseXpuKernel<device, functor<device, dtype>, bool, dtype>>(               \
            [](user_op::KernelComputeContext* ctx) { return functor<device, dtype>(); }, "out", \
            "in");                                                                              \
      })                                                                                        \
      .SetIsMatchedHob((user_op::HobDeviceType() == device)                                     \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

#define REGISTER_ISNAN_KERNEL(device, dtype) \
  REGISTER_UTIL_OPS_KERNELS(device, "isnan", dtype, IsNanFunctor)

#define REGISTER_ISINF_KERNEL(device, dtype) \
  REGISTER_UTIL_OPS_KERNELS(device, "isinf", dtype, IsInfFunctor)

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_UTIL_OPS_KERNELS_H_
