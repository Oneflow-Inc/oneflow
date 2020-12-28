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
#ifndef ONEFLOW_USER_KERNELS_ARG_WHERE_KERNEL_UTIL_H_
#define ONEFLOW_USER_KERNELS_ARG_WHERE_KERNEL_UTIL_H_

#include "oneflow/core/device/device_context.h"
#include "oneflow/core/common/shape_view.h"

namespace oneflow {

template<DeviceType device_type, typename IN_T, typename OUT_T, int NDIM>
struct ArgWhereKernelUtil {
  static void ArgWhere(DeviceCtx* ctx, const ShapeView& input_shape, const IN_T* input_ptr,
                       void* temp_storage, size_t temp_storage_bytes, OUT_T* output_ptr,
                       OUT_T* output_size_ptr);
  static size_t GetWorkspaceBytesSize(DeviceCtx* ctx, int64_t elem_cnt);
};

#define INSTANTIATE_ARG_WHERE_KERNEL_UTIL(device, itype, otype, ndim) \
  template struct ArgWhereKernelUtil<device, itype, otype, ndim>;

#define INSTANTIATE_ARG_WHERE_KERNEL_UTIL_WITH_DTYPE_PAIR(device, itype_pair, otype_pair, ndim) \
  INSTANTIATE_ARG_WHERE_KERNEL_UTIL(device, OF_PP_PAIR_FIRST(itype_pair),                       \
                                    OF_PP_PAIR_FIRST(otype_pair), ndim)

#define INSTANTIATE_ARG_WHERE_KERNEL_UTIL_FOR_DEVICE(device)                                    \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARG_WHERE_KERNEL_UTIL_WITH_DTYPE_PAIR, (device), \
                                   ARITHMETIC_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, DIM_SEQ)

}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_ARG_WHERE_KERNEL_UTIL_H_
