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
#include "oneflow/user/kernels/fold_kernel_util.h"
namespace oneflow {

namespace user_op {

// NDIM range: (1, 2, 3)
// SDIM range: (1, 2), 1 indicates channels_last, 2 indicates channels_first
template<typename T, typename INDEX_T, int NDIM, int SDIM>
struct FoldKernelUtil<DeviceType::kCPU, T, INDEX_T, NDIM, SDIM> {
  using ParamType = FoldParams<INDEX_T, NDIM, SDIM>;
  static void Forward(ep::Stream* stream, const void* raw_params, const T* input_ptr,
                      T* output_ptr) {
    const auto* params = static_cast<const ParamType*>(raw_params);
    for (INDEX_T in_offset = 0; in_offset < params->in_elem_cnt; ++in_offset) {
      using ParamType = FoldParams<INDEX_T, NDIM, SDIM>;
      INDEX_T in_index[ParamType::kInputNDim] = {0};
      INDEX_T out_index[ParamType::kOutputNDim] = {0};
      params->in_index_helper.OffsetToNdIndex(in_offset, in_index);
      if (!FoldIndexTransform<INDEX_T, NDIM, SDIM>(*params, in_index, out_index)) {
        INDEX_T out_offset = params->out_index_helper.NdIndexToOffset(out_index);
        XPUAdd<T>::Invoke(&input_ptr[in_offset], &output_ptr[out_offset]);
      } else {
        continue;
      }
    }
  }
};

INSTANTIATE_FOLD_KERNEL_UTIL_FOR_DEVICE(DeviceType::kCPU)

}  // namespace user_op

}  // namespace oneflow