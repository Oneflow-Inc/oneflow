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
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/max_unpool_kernel_util.h"
#include <cuda_fp16.h>

namespace oneflow {
namespace {

constexpr int kBlockSize = cuda::elementwise::kBlockSize;

int GetMinThreadNum(int64_t elem_num) { return std::min<int64_t>(elem_num, kBlockSize); }

int GetNumBlocks(int64_t elem_cnt) {
  int num_blocks = 0;
  OF_CUDA_CHECK(cuda::elementwise::GetNumBlocks(elem_cnt, &num_blocks));
  return num_blocks;
}

}  // namespace

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxUnpoolNdForward(const NdIndexOffsetHelper<IDX, 2> index_helper, IDX elem_num,
                                  const T* src, T* dest, const int64_t* indice_ptr,
                                  const int64_t y_hwd_size, const int64_t y_elem_num) {
  CUDA_1D_KERNEL_LOOP_T(IDX, num, elem_num) {
    IDX bc_idx, hwd_idx;
    index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
    IDX dest_idx = bc_idx * y_hwd_size + indice_ptr[num];
    if (dest_idx >= 0 && dest_idx < y_elem_num) { dest[dest_idx] = src[num]; }
  }
}

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxUnpoolNdBackward(const NdIndexOffsetHelper<IDX, 2> index_helper, IDX elem_num,
                                   const T* src, T* dest, const int64_t* indice_ptr,
                                   const int64_t dy_hwd_size, const int64_t dy_elem_num) {
  CUDA_1D_KERNEL_LOOP_T(IDX, num, elem_num) {
    IDX bc_idx, hwd_idx;
    index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
    IDX src_idx = bc_idx * dy_hwd_size + indice_ptr[num];
    if (src_idx >= 0 && src_idx < dy_elem_num) {
      dest[num] = src[src_idx];
    } else {
      dest[num] = 0.0f;
    }
  }
}

template<typename T, typename IDX>
struct UnpoolKernelUtil<DeviceType::kCUDA, T, IDX> {
  static void MaxUnpoolNdForward(ep::Stream* stream,
                                 const NdIndexOffsetHelper<IDX, 2>& index_helper, IDX elem_num,
                                 const T* src, T* dest, const int64_t* indice_ptr,
                                 const int64_t y_hwd_size, const int64_t y_elem_num) {
    DoCUDAMaxUnpoolNdForward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                       stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, y_hwd_size, y_elem_num);
  }

  static void MaxUnpoolNdBackward(ep::Stream* stream,
                                  const NdIndexOffsetHelper<IDX, 2>& index_helper, IDX elem_num,
                                  const T* src, T* dest, const int64_t* indice_ptr,
                                  const int64_t dy_hwd_size, const int64_t dy_elem_num) {
    DoCUDAMaxUnpoolNdBackward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                        stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, dy_hwd_size, dy_elem_num);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNPOOL_KERNEL_UTIL, (DeviceType::kCUDA),
                                 UNPOOL_DATA_TYPE_CUDA_SEQ, UNPOOL_IDX_DATA_TYPE_SEQ);
#if CUDA_VERSION >= 11000
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNPOOL_KERNEL_UTIL, (DeviceType::kCUDA),
                                 OF_PP_MAKE_TUPLE_SEQ(nv_bfloat16, DataType::kBFloat16),
                                 UNPOOL_IDX_DATA_TYPE_SEQ);
#endif  // CUDA_VERSION >= 11000

}  // namespace oneflow
#endif  // WITH_CUDA
