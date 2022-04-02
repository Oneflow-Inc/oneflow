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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/user/kernels/util_ops_kernels.h"

namespace oneflow {
namespace user_op {

template<typename T>
struct IsNanFunctor<DeviceType::kCPU, T, std::enable_if_t<std::is_floating_point<T>::value>> {
  OF_DEVICE_FUNC bool operator()(const T x) const { return std::isnan(x); }
};

template<typename T>
struct IsNanFunctor<DeviceType::kCPU, T, std::enable_if_t<!std::is_floating_point<T>::value>> {
  OF_DEVICE_FUNC bool operator()(const T x) const { return false; }
};

template<typename T>
struct IsInfFunctor<DeviceType::kCPU, T, std::enable_if_t<std::is_floating_point<T>::value>> {
  OF_DEVICE_FUNC bool operator()(const T x) const { return std::isinf(x); }
};

template<typename T>
struct IsInfFunctor<DeviceType::kCPU, T, std::enable_if_t<!std::is_floating_point<T>::value>> {
  OF_DEVICE_FUNC bool operator()(const T x) const { return false; }
};

#define REGISTER_UTIL_OPS_CPU_KERNEL(device, dtype_pair)      \
  REGISTER_ISNAN_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair)) \
  REGISTER_ISINF_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UTIL_OPS_CPU_KERNEL, (DeviceType::kCPU),
                                 UTIL_OPS_DATA_TYPE_SEQ);

}  // namespace user_op
}  // namespace oneflow
