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
// #include "oneflow/core/ep/include/stream.h"
#ifdef WITH_CUDA
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/user/kernels/max_unpool_kernel_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
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
                                  const int64_t y_hwd_size) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX bc_idx, hwd_idx;
    index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
    IDX dest_idx = bc_idx * y_hwd_size + indice_ptr[num];
    dest[dest_idx] = src[num];
  }
}

template<typename T, typename IDX>
__launch_bounds__(kBlockSize) __global__
    void DoCUDAMaxUnpoolNdBackward(const NdIndexOffsetHelper<IDX, 2> index_helper, IDX elem_num,
                                   const T* src, T* dest, const int64_t* indice_ptr,
                                   const int64_t dx_hwd_size) {
  XPU_1D_KERNEL_LOOP(num, elem_num) {
    IDX bc_idx, hwd_idx;
    index_helper.OffsetToNdIndex(num, bc_idx, hwd_idx);
    IDX dest_idx = bc_idx * dx_hwd_size + indice_ptr[num];
    dest[dest_idx] = src[num];
  }
}

template<typename T, typename IDX>
struct UnpoolKernelUtil<DeviceType::kCUDA, T, IDX> {
  static void MaxUnpoolNdForward(ep::Stream* stream,
                                 const NdIndexOffsetHelper<IDX, 2>& index_helper, IDX elem_num,
                                 const T* src, T* dest, const int64_t* indice_ptr,
                                 const int64_t y_hwd_size) {
    DoCUDAMaxUnpoolNdForward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                       stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, y_hwd_size);
  }

  static void MaxUnpoolNdBackward(ep::Stream* stream,
                                  const NdIndexOffsetHelper<IDX, 2>& index_helper, IDX elem_num,
                                  const T* src, T* dest, const int64_t* indice_ptr,
                                  const int64_t dx_hwd_size) {
    DoCUDAMaxUnpoolNdForward<T, IDX><<<GetNumBlocks(elem_num), GetMinThreadNum(elem_num), 0,
                                       stream->As<ep::CudaStream>()->cuda_stream()>>>(
        index_helper, elem_num, src, dest, indice_ptr, dx_hwd_size);
  }
};

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INSTANTIATE_UNPOOL_KERNEL_UTIL, (DeviceType::kCUDA),
                                 UNPOOL_DATA_TYPE_CUDA_SEQ, UNPOOL_IDX_DATA_TYPE_SEQ);

}  // namespace oneflow
#endif  // WITH_CUDA
