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
#ifdef WITH_CUDA

#include "oneflow/core/framework/framework.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/fold_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"

namespace oneflow {

namespace user_op {

namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetNumBlocks(int64_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

// NDIM range: (1, 2, 3)
// SDIM range: (1, 2), 1 indicates channels_last, 2 indicates channels_first
template<typename T, typename INDEX_T, int NDIM, int SDIM>
__global__ void CudaFoldForward(FoldParams<INDEX_T, NDIM, SDIM> params, const T* input_ptr,
                                T* output_ptr) {
  CUDA_1D_KERNEL_LOOP_T(INDEX_T, in_offset, params.in_elem_cnt) {
    using ParamType = FoldParams<INDEX_T, NDIM, SDIM>;
    INDEX_T in_index[ParamType::kInputNDim] = {0};
    INDEX_T out_index[ParamType::kOutputNDim] = {0};
    params.in_index_helper.OffsetToNdIndex(in_offset, in_index);
    if (!FoldIndexTransform<INDEX_T, NDIM, SDIM>(params, in_index, out_index)) {
      INDEX_T out_offset = params.out_index_helper.NdIndexToOffset(out_index);
      XPUAdd<T>::Invoke(&input_ptr[in_offset], &output_ptr[out_offset]);
    } else {
      continue;
    }
  }
}

}  // namespace

template<typename T, typename INDEX_T, int NDIM, int SDIM>
struct FoldKernelUtil<DeviceType::kCUDA, T, INDEX_T, NDIM, SDIM> {
  using ParamType = FoldParams<INDEX_T, NDIM, SDIM>;
  static void Forward(ep::Stream* stream, const void* raw_params, const T* input_ptr,
                      T* output_ptr) {
    const auto* fold_params = static_cast<const ParamType*>(raw_params);
    CudaFoldForward<T, INDEX_T, NDIM, SDIM>
        <<<GetNumBlocks(fold_params->in_elem_cnt), kBlockSize, 0,
           stream->As<ep::CudaStream>()->cuda_stream()>>>(*fold_params, input_ptr, output_ptr);
  }
};

INSTANTIATE_FOLD_KERNEL_UTIL_FOR_DEVICE(DeviceType::kCUDA)

}  // namespace user_op
}  // namespace oneflow
#endif  // WITH_CUDA