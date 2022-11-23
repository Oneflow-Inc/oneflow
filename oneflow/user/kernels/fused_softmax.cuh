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

#ifndef ONEFLOW_USER_KERNELS_FUSED_SOFTMAX_H_
#define ONEFLOW_USER_KERNELS_FUSED_SOFTMAX_H_

#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {
namespace cuda {
namespace fused_softmax {

inline void SimplifyBroadcastDims(size_t num_a_dims, const int64_t* a_dims, size_t num_b_dims,
                                  const int64_t* b_dims, size_t* simplified_num_dims,
                                  int64_t* simplified_a_dims, int64_t* simplified_b_dims) {
  const size_t num_max_dims = std::max(num_a_dims, num_b_dims);
  auto MakeGetDim = [num_max_dims](size_t num_dims, const int64_t* dims) {
    const int64_t num_padding_dims = num_max_dims - num_dims;
    return [num_padding_dims, dims](size_t index) {
      return index < num_padding_dims ? 1 : dims[index - num_padding_dims];
    };
  };
  auto GetADim = MakeGetDim(num_a_dims, a_dims);
  auto GetBDim = MakeGetDim(num_b_dims, b_dims);
  *simplified_num_dims = 0;
  bool prev_broadcast_a = false;
  bool prev_broadcast_b = false;
  for (int64_t i = 0; i < num_max_dims; ++i) {
    const int64_t a_dim = GetADim(i);
    const int64_t b_dim = GetBDim(i);
    const int64_t broadcast_dim = std::max(a_dim, b_dim);
    CHECK_GT(broadcast_dim, 0);
    const bool broadcast_a = (a_dim == 1);
    const bool broadcast_b = (b_dim == 1);
    CHECK((a_dim == broadcast_dim) || broadcast_a);
    CHECK((b_dim == broadcast_dim) || broadcast_b);
    if (broadcast_dim == 1) {
      continue;
    } else if (*simplified_num_dims != 0
               && (prev_broadcast_a == broadcast_a && prev_broadcast_b == broadcast_b)) {
      simplified_a_dims[*simplified_num_dims - 1] *= a_dim;
      simplified_b_dims[*simplified_num_dims - 1] *= b_dim;
    } else {
      simplified_a_dims[*simplified_num_dims] = a_dim;
      simplified_b_dims[*simplified_num_dims] = b_dim;
      *simplified_num_dims += 1;
      prev_broadcast_a = broadcast_a;
      prev_broadcast_b = broadcast_b;
    }
  }
}

template<size_t num_dims, typename IndexType>
struct BroadcastMaskSoftmaxParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> mask_index_helper;
  const int64_t* mask_dims{};
  int64_t row_size;
  float fill;
  float scale;
};

struct ElementwiseMaskSoftmaxParams {
  int64_t row_size;
  float fill;
  float scale;
};

template<typename SRC, typename DST, typename MASK, size_t num_dims, typename IndexType>
struct BroadcastScaleMaskLoad {
  BroadcastScaleMaskLoad(const SRC* src, const MASK* mask,
                         BroadcastMaskSoftmaxParams<num_dims, IndexType> params)
      : src(src), mask(mask), params(params) {
    for (int i = 0; i < num_dims; i++) { mask_dims[i] = params.mask_dims[i]; }
  }
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) {
    cuda::softmax::Pack<SRC, N> pack;
    cuda::softmax::Pack<MASK, N> mask_pack;
    const IndexType offset = row * params.row_size + col;
    IndexType input_index[num_dims];
    IndexType mask_index[num_dims];
    params.src_index_helper.OffsetToNdIndex(offset, input_index);
    for (int dim = 0; dim < num_dims; ++dim) {
      if (mask_dims[dim] == 1) {
        mask_index[dim] = 0;
      } else {
        mask_index[dim] = input_index[dim];
      }
    }
    const IndexType mask_offset = params.mask_index_helper.NdIndexToOffset(mask_index);
    pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src) + offset / N);
    mask_pack.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<MASK, N>*>(mask) + mask_offset / N);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        dst[i] = static_cast<DST>(params.fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(params.scale);
      }
    }
  }
  const SRC* src;
  const MASK* mask;
  int64_t mask_dims[num_dims];
  BroadcastMaskSoftmaxParams<num_dims, IndexType> params;
};

template<typename SRC, typename DST, typename MASK>
struct ElementwiseScaleMaskLoad {
  ElementwiseScaleMaskLoad(const SRC* src, const MASK* mask, ElementwiseMaskSoftmaxParams param)
      : src(src), mask(mask), param(param) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) {
    cuda::softmax::Pack<SRC, N> pack;
    const int64_t offset = (row * param.row_size + col) / N;
    pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src) + offset);
    cuda::softmax::Pack<int8_t, N> mask_pack;
    mask_pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<MASK, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        dst[i] = static_cast<DST>(param.fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(param.scale);
      }
    }
  }
  const SRC* src;
  const MASK* mask;
  ElementwiseMaskSoftmaxParams param;
};

template<typename SRC, typename DST, typename MASK, size_t num_dims, typename IndexType>
struct BroadcastScaleMaskStore {
  BroadcastScaleMaskStore(DST* dst, const MASK* mask,
                          BroadcastMaskSoftmaxParams<num_dims, IndexType> params)
      : dst(dst), mask(mask), params(params) {
    for (int i = 0; i < num_dims; ++i) { mask_dims[i] = params.mask_dims[i]; }
  }
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> pack;
    cuda::softmax::Pack<MASK, N> mask_pack;
    const IndexType offset = row * params.row_size + col;
    IndexType input_index[num_dims];
    IndexType mask_index[num_dims];
    params.src_index_helper.OffsetToNdIndex(offset, input_index);
    for (int dim = 0; dim < num_dims; ++dim) {
      if (mask_dims[dim] == 1) {
        mask_index[dim] = 0;
      } else {
        mask_index[dim] = input_index[dim];
      }
    }
    const IndexType mask_offset = params.mask_index_helper.NdIndexToOffset(mask_index);
    mask_pack.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<MASK, N>*>(mask) + mask_offset / N);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        pack.elem[i] = static_cast<DST>(params.fill);
      } else {
        pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(params.scale);
      }
    }
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst) + offset / N) = pack.storage;
  }
  DST* dst;
  const MASK* mask;
  int64_t mask_dims[num_dims];
  BroadcastMaskSoftmaxParams<num_dims, IndexType> params;
};

template<typename SRC, typename DST, typename MASK>
struct ElementwiseScaleMaskStore {
  ElementwiseScaleMaskStore(DST* dst, const MASK* mask, ElementwiseMaskSoftmaxParams params)
      : dst(dst), mask(mask), params(params) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> pack;
    const int64_t offset = (row * params.row_size + col) / N;
    cuda::softmax::Pack<MASK, N> mask_pack;
    mask_pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<MASK, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        pack.elem[i] = params.fill;
      } else {
        pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(params.scale);
      }
    }
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  const MASK* mask;
  ElementwiseMaskSoftmaxParams params;
};

template<typename SRC, typename DST>
struct MaskScaleLoad {
  MaskScaleLoad(const SRC* src, const bool* mask, int64_t row_size, SRC scale)
      : src(src), mask(mask), row_size(row_size), scale(scale) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src) + offset);
    cuda::softmax::Pack<bool, N> mask_pack;
    mask_pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<bool, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(mask_pack.elem[i])
               * static_cast<DST>(scale);
    }
  }
  const SRC* src;
  const bool* mask;
  int64_t row_size;
  SRC scale;
};

template<typename SRC, typename DST>
struct DropoutStore {
  DropoutStore(DST* dst, DST* softmax_y, const bool* mask, int64_t row_size, DST scale)
      : dst(dst), softmax_y(softmax_y), mask(mask), row_size(row_size), scale(scale) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> softmax_y_pack;
    cuda::softmax::Pack<DST, N> dst_pack;
    const int64_t offset = (row * row_size + col) / N;
    cuda::softmax::Pack<bool, N> mask_pack;
    mask_pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<bool, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      softmax_y_pack.elem[i] = static_cast<DST>(src[i]);
      dst_pack.elem[i] =
          static_cast<DST>(src[i]) * static_cast<DST>(mask_pack.elem[i]) * static_cast<DST>(scale);
    }
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(softmax_y) + offset) =
        softmax_y_pack.storage;
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst) + offset) = dst_pack.storage;
  }
  DST* dst;
  DST* softmax_y;
  const bool* mask;
  int64_t row_size;
  DST scale;
};

}  // namespace fused_softmax
}  // namespace cuda
}  // namespace oneflow

#endif  // ONEFLOW_USER_KERNELS_FUSED_SOFTMAX_H_
