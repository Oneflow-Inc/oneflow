#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow{

namespace fused_scale_mask_softmax{

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
    mask_pack.storage =
      *(reinterpret_cast<const cuda::softmax::PackType<MASK, N>*>(mask) + offset);
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
  BroadcastScaleMaskStore(DST* dst, const MASK* mask, BroadcastMaskSoftmaxParams<num_dims, IndexType> params)
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
    mask_pack.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<MASK, N>*>(mask) + offset);
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


}

}