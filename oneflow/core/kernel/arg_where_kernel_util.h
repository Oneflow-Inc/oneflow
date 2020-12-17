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
#ifndef ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_
#define ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/shape_view.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename I, size_t NDims>
struct ArgWhereKernelUtil {
  static void ArgWhere(DeviceCtx* ctx, const ShapeView& in_shape, const T* in_ptr, void* tmp,
                       size_t tmp_max_bytes, I* out_ptr, I* out_size_ptr);
  static size_t GetArgWhereWorkspaceSizeInBytes(DeviceCtx* ctx, int64_t n);
};

#define INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, dtype, itype, ndims) \
  template struct ArgWhereKernelUtil<device_type_v, dtype, itype, ndims>;

#define INSTANTIATE_ARG_WHERE_KERNEL_UTIL(device_type_v, dtype_pair, itype_pair)          \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 1)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 2)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 3)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 4)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 5)             \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL_INTERNAL(device_type_v, OF_PP_PAIR_FIRST(dtype_pair), \
                                             OF_PP_PAIR_FIRST(itype_pair), 6)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_KERNEL_ARG_WHERE_KERNEL_UTIL_H_
