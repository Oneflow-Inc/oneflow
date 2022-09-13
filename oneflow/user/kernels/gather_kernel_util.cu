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
#include "oneflow/user/kernels/gather_kernel_util.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/nd_index_offset_helper.h"
#include <assert.h>
#if CUDA_VERSION >= 11000
#include <cuda_bf16.h>
#endif  // CUDA_VERSION >= 11000
namespace oneflow {

namespace {

template<typename T, typename K, typename IDX>
__global__ void GatherForwardGpu(const IDX elem_cnt, NdIndexOffsetHelper<IDX, 3> in_helper,
                                 NdIndexOffsetHelper<IDX, 3> out_helper, const K* indices,
                                 const T* in, const IDX gather_dim_size, T* out, const IDX offset) {
  IDX index[3];
  CUDA_1D_KERNEL_LOOP_T(IDX, i, elem_cnt) {
    out_helper.OffsetToNdIndex(i, index);
    index[1] = indices[index[1]] - offset;
    T v{};
    if (index[1] >= 0 && index[1] < gather_dim_size) { v = in[in_helper.NdIndexToOffset(index)]; }
    out[i] = v;
  }
}

bool IsSafeUseIndex32(int64_t outer_dim_size, int64_t gather_dim_size, int64_t inner_dim_size,
                      int64_t num_indices) {
  const int64_t in_elem_cnt = outer_dim_size * gather_dim_size * inner_dim_size;
  const int64_t out_elem_cnt = outer_dim_size * num_indices * inner_dim_size;
  return std::max(out_elem_cnt, in_elem_cnt) < GetMaxVal<int32_t>() / 2;
}

template<typename T, typename K>
void DispatchIndexSize(ep::Stream* stream, int64_t outer_dim_size, int64_t gather_dim_size,
                       int64_t inner_dim_size, int64_t num_indices, int64_t offset,
                       const K* indices, const T* in, T* out) {
  const int64_t out_elem_cnt = outer_dim_size * num_indices * inner_dim_size;
  if (IsSafeUseIndex32(outer_dim_size, gather_dim_size, inner_dim_size, num_indices)) {
    NdIndexOffsetHelper<int32_t, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
    NdIndexOffsetHelper<int32_t, 3> out_helper(outer_dim_size, num_indices, inner_dim_size);
    GatherForwardGpu<T, K, int32_t><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock,
                                      0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        out_elem_cnt, in_helper, out_helper, indices, in, gather_dim_size, out, offset);
  } else {
    NdIndexOffsetHelper<int64_t, 3> in_helper(outer_dim_size, gather_dim_size, inner_dim_size);
    NdIndexOffsetHelper<int64_t, 3> out_helper(outer_dim_size, num_indices, inner_dim_size);
    GatherForwardGpu<T, K, int64_t><<<BlocksNum4ThreadsNum(out_elem_cnt), kCudaThreadsNumPerBlock,
                                      0, stream->As<ep::CudaStream>()->cuda_stream()>>>(
        out_elem_cnt, in_helper, out_helper, indices, in, gather_dim_size, out, offset);
  }
}

template<typename K, typename T>
bool TryDispatchMovementType(ep::Stream* stream, int64_t outer_dim_size, int64_t gather_dim_size,
                             int64_t inner_dim_size, int64_t num_indices, int64_t offset,
                             const K* indices, const void* in, void* out) {
  if (reinterpret_cast<uintptr_t>(in) % sizeof(T) == 0
      && reinterpret_cast<uintptr_t>(out) % sizeof(T) == 0 && inner_dim_size % sizeof(T) == 0) {
    DispatchIndexSize<T, K>(stream, outer_dim_size, gather_dim_size, inner_dim_size / sizeof(T),
                            num_indices, offset, indices, static_cast<const T*>(in),
                            static_cast<T*>(out));
    return true;
  } else {
    return false;
  }
}

template<typename K>
void DispatchMovementSize(ep::Stream* stream, int64_t outer_dim_size, int64_t gather_dim_size,
                          int64_t inner_dim_size, int64_t num_indices, int64_t offset,
                          const K* indices, const void* in, void* out) {
  using Func = bool (*)(ep::Stream * stream, int64_t outer_dim_size, int64_t gather_dim_size,
                        int64_t inner_dim_size, int64_t num_indices, int64_t offset,
                        const K* indices, const void* in, void* out);
  Func funcs[] = {
      TryDispatchMovementType<K, ulonglong2>,  // 16B
      TryDispatchMovementType<K, uint64_t>,    // 8B
      TryDispatchMovementType<K, uint32_t>,    // 4B
      TryDispatchMovementType<K, uint16_t>,    // 2B
      TryDispatchMovementType<K, uint8_t>,     // 1B
  };
  for (size_t i = 0; i < sizeof(funcs) / sizeof(funcs[0]); ++i) {
    if (funcs[i](stream, outer_dim_size, gather_dim_size, inner_dim_size, num_indices, offset,
                 indices, in, out)) {
      break;
    }
  }
}

}  // namespace

template<typename T, typename K>
struct GatherKernelUtilImpl<DeviceType::kCUDA, T, K> final {
  static void Forward(ep::Stream* stream, const K* indices, int64_t num_indices, const T* in,
                      const Shape& flat_in_shape, T* out, const int64_t offset) {
    DispatchMovementSize(stream, flat_in_shape.At(0), flat_in_shape.At(1),
                         flat_in_shape.At(2) * sizeof(T), num_indices, offset, indices, in, out);
  }
};

#define INITIATE_GATHER_KERNEL_UTIL_CUDA_IMPL(in_type_pair, index_type_pair)              \
  template struct GatherKernelUtilImpl<DeviceType::kCUDA, OF_PP_PAIR_FIRST(in_type_pair), \
                                       OF_PP_PAIR_FIRST(index_type_pair)>;
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_CUDA_IMPL,
                                 GATHER_DATA_TYPE_SEQ HALF_DATA_TYPE_SEQ, GATHER_INDEX_TYPE_SEQ);
#if CUDA_VERSION >= 11000
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(INITIATE_GATHER_KERNEL_UTIL_CUDA_IMPL,
                                 OF_PP_MAKE_TUPLE_SEQ(nv_bfloat16, DataType::kBFloat16),
                                 GATHER_INDEX_TYPE_SEQ);
#endif
#undef INITIATE_GATHER_KERNEL_UTIL_CUDA_IMPL

}  // namespace oneflow
