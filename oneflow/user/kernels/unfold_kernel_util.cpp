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
#include "oneflow/user/kernels/unfold_kernel_util.h"

namespace oneflow {

namespace user_op {

// NDIM range: (1, 2, 3)
// SDIM range: (1, 2), 1 indicates channels_last, 2 indicates channels_first
template<typename T, typename INDEX_T, int NDIM, int SDIM>
struct UnfoldKernelUtil<DeviceType::kCPU, T, INDEX_T, NDIM, SDIM> {
  using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
  static void Forward(ep::Stream* stream, const UnfoldParams<INDEX_T, NDIM, SDIM>* raw_params,
                      const T* input_ptr, T* output_ptr) {
    for (INDEX_T out_offset = 0; out_offset < raw_params->out_elem_cnt; ++out_offset) {
      using ParamType = UnfoldParams<INDEX_T, NDIM, SDIM>;
      INDEX_T in_index[ParamType::kInputNDim] = {0};
      INDEX_T out_index[ParamType::kOutputNDim] = {0};
      raw_params->out_index_helper.OffsetToNdIndex(out_offset, out_index);
      if (!UnfoldIndexTransform<INDEX_T, NDIM, SDIM>(*raw_params, out_index, in_index)) {
        INDEX_T in_offset = raw_params->in_index_helper.NdIndexToOffset(in_index);
        output_ptr[out_offset] = input_ptr[in_offset];
      } else {
        output_ptr[out_offset] = static_cast<T>(kUnfoldPaddingValue);
      }
    }
  }
};

INSTANTIATE_UNFOLD_KERNEL_UTIL_FOR_DEVICE(DeviceType::kCPU)

}  // namespace user_op

}  // namespace oneflow