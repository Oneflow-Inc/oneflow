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
#ifndef ONEFLOW_USER_KERNELS_ISNAN_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_ISNAN_KERNEL_UTIL_H_

#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"

namespace oneflow {
namespace user_op {

#define UTIL_OPS_FUNCTOR_DTYPE_SEQ \
  FLOATING_DATA_TYPE_SEQ           \
  INT_DATA_TYPE_SEQ                \
  UNSIGNED_INT_DATA_TYPE_SEQ       \
  BOOL_DATA_TYPE_SEQ

template<DeviceType device_type, typename T>
struct IsNanFunctor {
  void operator()(ep::Stream*, bool*, const T*, const size_t);
};

template<DeviceType device_type, typename T>
struct IsInfFunctor {
  void operator()(ep::Stream*, bool*, const T*, const size_t);
};

#define INSTANTIATE_Util_OPS_FUNCTOR(device, dtype)              \
  template struct IsNanFunctor<device, OF_PP_PAIR_FIRST(dtype)>; \
  template struct IsInfFunctor<device, OF_PP_PAIR_FIRST(dtype)>;

}  // namespace user_op
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_ISNAN_KERNEL_UTIL_H_
