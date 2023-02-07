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
#include "oneflow/user/kernels/arg_where_kernel_util.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include "oneflow/core/common/small_vector.h"
#include "oneflow/core/kernel/kernel_util.h"

namespace oneflow {

template<typename IN_T, typename OUT_T, int NDIM>
struct ArgWhereKernelUtil<DeviceType::kCPU, IN_T, OUT_T, NDIM> {
  static void ArgWhere(ep::Stream* stream, const ShapeView& input_shape, const IN_T* input_ptr,
                       void* temp_storage, size_t temp_storage_bytes, OUT_T* output_ptr,
                       OUT_T* output_size_ptr) {
    // deal with empty blob
    if (input_shape.elem_cnt() == 0) {
      Memset<DeviceType::kCPU>(stream, output_size_ptr, 0, sizeof(OUT_T));
      return;
    }

    const int64_t elem_cnt = input_shape.elem_cnt();
    CHECK_LE(elem_cnt, std::numeric_limits<OUT_T>::max());
    OUT_T true_cnt = 0;
    OUT_T dims[NDIM] = {0};
    std::transform(input_shape.ptr(), input_shape.ptr() + input_shape.NumAxes(), dims,
                   [](int64_t dim) { return static_cast<OUT_T>(dim); });
    NdIndexOffsetHelper<OUT_T, NDIM> index_converter(dims);
    FOR_RANGE(int64_t, i, 0, elem_cnt) {
      if (static_cast<bool>(input_ptr[i])) {
        index_converter.OffsetToNdIndex(i, output_ptr + true_cnt * NDIM);
        true_cnt += 1;
      }
    }
    *output_size_ptr = true_cnt;
  }

  static size_t GetWorkspaceBytesSize(ep::Stream* stream, int64_t elem_cnt) { return 0; }
};

INSTANTIATE_ARG_WHERE_KERNEL_UTIL_FOR_DEVICE(DeviceType::kCPU)

#define INSTANTIATE_CPU_FLOAT16_ARG_WHERE_KERNEL_UTIL                                              \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_ARG_WHERE_KERNEL_UTIL_WITH_DTYPE_PAIR,              \
                                   (DeviceType::kCPU), FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ, \
                                   DIM_SEQ)

INSTANTIATE_CPU_FLOAT16_ARG_WHERE_KERNEL_UTIL

template<DeviceType device_type, typename IN_T, typename OUT_T>
void SetOutputSize(ep::Stream* stream, const IN_T* input_ptr, OUT_T* output_size_ptr) {
  if (*input_ptr == GetZeroVal<IN_T>()) {
    *output_size_ptr = GetZeroVal<OUT_T>();
  } else {
    *output_size_ptr = GetOneVal<OUT_T>();
  }
}

INSTANTIATE_SET_OUTPUT_SIZE_FOR_DEVICE(DeviceType::kCPU)

#define INSTANTIATE_CPU_FLOAT16_SET_OUTPUT_SIZE                                 \
  OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_SET_OUTPUT_SIZE_WITH_DTYPE_PAIR, \
                                   (DeviceType::kCPU), FLOAT16_DATA_TYPE_SEQ, INDEX_DATA_TYPE_SEQ)

INSTANTIATE_CPU_FLOAT16_SET_OUTPUT_SIZE

}  // namespace oneflow
