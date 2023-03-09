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
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/user/kernels/fused_softmax.cuh"

namespace oneflow {
namespace cuda {

namespace {

template<typename IndexType, size_t NDIM>
struct BroadcastMapper {
  using index_type = IndexType;
  IndexType src_dims[NDIM] = {0};
  IndexType dst_dims[NDIM] = {0};

  template<typename DimType>
  BroadcastMapper(const DimType* arg_src_dims, const DimType* arg_dst_dims) {
    for (size_t i = 0; i < NDIM; ++i) { src_dims[i] = arg_src_dims[i]; }
    for (size_t i = 0; i < NDIM; ++i) { dst_dims[i] = arg_dst_dims[i]; }
  }

  __device__ IndexType map(IndexType src) const {
    NdIndexOffsetHelper<IndexType, NDIM> src_index_helper(src_dims);
    NdIndexOffsetHelper<IndexType, NDIM> dst_index_helper(dst_dims);
    IndexType src_index[NDIM];
    IndexType dst_index[NDIM];
    src_index_helper.OffsetToNdIndex(src, src_index);
#pragma unroll
    for (int dim = 0; dim < NDIM; ++dim) {
      if (dst_dims[dim] == 1) {
        dst_index[dim] = 0;
      } else {
        dst_index[dim] = src_index[dim];
      }
    }
    return dst_index_helper.NdIndexToOffset(dst_index);
  }
};

template<typename IndexType>
struct ElementwiseMapper {
  using index_type = IndexType;
  ElementwiseMapper() {}
  __device__ IndexType map(IndexType index) const { return index; }
};

template<typename SRC, typename DST, typename MASK, typename BiasMapper, typename MaskMapper>
struct BiasAddScaleMaskLoad {
  static_assert(
      std::is_same<typename BiasMapper::index_type, typename MaskMapper::index_type>::value, "");
  using IndexType = typename BiasMapper::index_type;
  const SRC* src;
  const SRC* bias;
  const MASK* mask;
  const DST fill;
  const DST scale;
  const IndexType row_size;
  const BiasMapper bias_mapper;
  const MaskMapper mask_mapper;

  BiasAddScaleMaskLoad(const SRC* src, const SRC* bias, const MASK* mask, const DST fill,
                       const DST scale, const IndexType row_size, const BiasMapper bias_mapper,
                       const MaskMapper mask_mapper)
      : src(src),
        bias(bias),
        mask(mask),
        fill(fill),
        scale(scale),
        row_size(row_size),
        bias_mapper(bias_mapper),
        mask_mapper(mask_mapper) {}

  template<int N>
  __device__ void load(DST* dst, IndexType row, IndexType col) {
    softmax::Pack<SRC, N> src_pack;
    softmax::Pack<SRC, N> bias_pack;
    softmax::Pack<MASK, N> mask_pack;
    const IndexType offset = row * row_size + col;
    const IndexType bias_offset = bias_mapper.map(offset);
    const IndexType mask_offset = mask_mapper.map(offset);
    src_pack.storage = *(reinterpret_cast<const softmax::PackType<SRC, N>*>(src) + offset / N);
    bias_pack.storage =
        *(reinterpret_cast<const softmax::PackType<SRC, N>*>(bias) + bias_offset / N);
    mask_pack.storage =
        *(reinterpret_cast<const softmax::PackType<MASK, N>*>(mask) + mask_offset / N);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        dst[i] = fill;
      } else {
        dst[i] = static_cast<DST>(src_pack.elem[i] + bias_pack.elem[i]) * scale;
      }
    }
  }
};

template<typename T, typename MASK>
void DispatchForward(cudaStream_t stream, const user_op::Tensor* x, const user_op::Tensor* bias,
                     const user_op::Tensor* mask, const user_op::Tensor* dropout_mask,
                     const float mask_fill, const float scale, const float dropout_scale,
                     user_op::Tensor* y, user_op::Tensor* softmax_y) {
  using ComputeType = typename softmax::DefaultComputeType<T>::type;
  using IndexType = int32_t;
  constexpr int kMaxNDim = 5;

  const auto& x_shape = x->shape_view();
  CHECK_GE(x_shape.size(), 2);
  // the last dim is softmax dim which is considered as col
  int64_t ncol = x_shape[x_shape.size() - 1];
  int64_t nrow = x_shape.elem_cnt() / ncol;
  fused_softmax::DropoutStore<ComputeType, T> store(
      y->mut_dptr<T>(), softmax_y->mut_dptr<T>(), dropout_mask->dptr<bool>(), ncol, dropout_scale);

  size_t bias_sndim = 0;
  int64_t bias_x_sdims[kMaxNDim];
  int64_t bias_sdims[kMaxNDim];
  const auto& bias_shape = bias->shape_view();
  fused_softmax::SimplifyBroadcastDims(x_shape.size(), x_shape.ptr(), bias_shape.size(),
                                       bias_shape.ptr(), &bias_sndim, bias_x_sdims, bias_sdims);
  size_t mask_sndim = 0;
  int64_t mask_x_sdims[kMaxNDim];
  int64_t mask_sdims[kMaxNDim];
  const auto& mask_shape = mask->shape_view();
  fused_softmax::SimplifyBroadcastDims(x_shape.size(), x_shape.ptr(), mask_shape.size(),
                                       mask_shape.ptr(), &mask_sndim, mask_x_sdims, mask_sdims);

#define DISPATCH_BIAS_ADD_SCALE_MASK_SOFTMAX(bias_mapper, mask_mapper)                           \
  BiasAddScaleMaskLoad<T, ComputeType, MASK, decltype(bias_mapper), decltype(mask_mapper)> load( \
      x->dptr<T>(), bias->dptr<T>(), mask->dptr<MASK>(), mask_fill, scale, ncol, bias_mapper,    \
      mask_mapper);                                                                              \
  OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(   \
      stream, load, store, nrow, ncol)))

  if (bias_sndim == 1 && mask_sndim == 1) {
    // bias elementwise
    // mask elementwise
    ElementwiseMapper<IndexType> bias_mapper;
    ElementwiseMapper<IndexType> mask_mapper;
    DISPATCH_BIAS_ADD_SCALE_MASK_SOFTMAX(bias_mapper, mask_mapper);
  } else if (bias_sndim == 1 && mask_sndim == 2) {
    // bias elementwise
    // mask broadcast: (M, 1) -> (M, N) or (1, N) -> (M, N)
    ElementwiseMapper<IndexType> bias_mapper;
    BroadcastMapper<IndexType, 2> mask_mapper(mask_x_sdims, mask_sdims);
    DISPATCH_BIAS_ADD_SCALE_MASK_SOFTMAX(bias_mapper, mask_mapper);
  } else if (bias_sndim == 1 && mask_sndim == 3) {
    // bias elementwise
    // mask broadcast: (M, 1, N) -> (M, K, N)
    ElementwiseMapper<IndexType> bias_mapper;
    BroadcastMapper<IndexType, 3> mask_mapper(mask_x_sdims, mask_sdims);
    DISPATCH_BIAS_ADD_SCALE_MASK_SOFTMAX(bias_mapper, mask_mapper);
  } else if (bias_sndim == 2 && mask_sndim == 1) {
    // bias broadcast: (M, 1) -> (M, N) or (1, N) -> (M, N)
    // mask elementwise
    BroadcastMapper<IndexType, 2> bias_mapper(bias_x_sdims, bias_sdims);
    ElementwiseMapper<IndexType> mask_mapper;
    DISPATCH_BIAS_ADD_SCALE_MASK_SOFTMAX(bias_mapper, mask_mapper);
  } else if (bias_sndim == 2 && mask_sndim == 2) {
    // bias broadcast: (M, 1) -> (M, N) or (1, N) -> (M, N)
    // mask broadcast: (M, 1) -> (M, N) or (1, N) -> (M, N)
    BroadcastMapper<IndexType, 2> bias_mapper(bias_x_sdims, bias_sdims);
    BroadcastMapper<IndexType, 2> mask_mapper(mask_x_sdims, mask_sdims);
    DISPATCH_BIAS_ADD_SCALE_MASK_SOFTMAX(bias_mapper, mask_mapper);
  } else if (bias_sndim == 2 && mask_sndim == 3) {
    // bias broadcast: (M, 1) -> (M, N) or (1, N) -> (M, N)
    // mask broadcast: (M, 1, N) -> (M, K, N)
    BroadcastMapper<IndexType, 2> bias_mapper(bias_x_sdims, bias_sdims);
    BroadcastMapper<IndexType, 3> mask_mapper(mask_x_sdims, mask_sdims);
    DISPATCH_BIAS_ADD_SCALE_MASK_SOFTMAX(bias_mapper, mask_mapper);
    // not support for now
    // } else if (bias_sndim == 3 && mask_sndim == 1) {
    // } else if (bias_sndim == 3 && mask_sndim == 2) {
    // } else if (bias_sndim == 3 && mask_sndim == 3) {
  } else {
    UNIMPLEMENTED() << ", bias_sndim=" << bias_sndim << ", mask_sndim=" << mask_sndim;
  }

#undef DISPATCH_BIAS_ADD_SCALE_MASK_SOFTMAX
}

template<typename T, typename MASK>
class FusedBiasAddScaleMaskSoftmaxDropoutKernel final : public user_op::OpKernel {
 public:
  FusedBiasAddScaleMaskSoftmaxDropoutKernel() = default;
  ~FusedBiasAddScaleMaskSoftmaxDropoutKernel() override = default;

 private:
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }

  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* bias = ctx->Tensor4ArgNameAndIndex("bias", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const user_op::Tensor* dropout_mask = ctx->Tensor4ArgNameAndIndex("dropout_mask", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* softmax_y = ctx->Tensor4ArgNameAndIndex("softmax_y", 0);

    const float mask_fill = ctx->Attr<float>("mask_fill_value");
    const float scale = ctx->Attr<float>("scale_value");
    const float dropout_scale = ctx->Attr<float>("dropout_scale_value");

    const ShapeView& x_shape = x->shape_view();
    // int32 index computing is much faster than int64
    // TODO: consider using multiple int32 computing to substitute int64 computing
    CHECK_LT(x_shape.elem_cnt(), INT_MAX) << "only support int32 max limits size of elements";
    DispatchForward<T, MASK>(ctx->stream()->As<ep::CudaStream>()->cuda_stream(), x, bias, mask,
                             dropout_mask, mask_fill, scale, dropout_scale, y, softmax_y);
  }
};

}  // namespace

#define REGISTER_FUSED_BIAS_ADD_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL(dtype, mask_dtype) \
  REGISTER_USER_KERNEL("fused_bias_add_scale_mask_softmax_dropout")                       \
      .SetCreateFn<FusedBiasAddScaleMaskSoftmaxDropoutKernel<dtype, mask_dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                    \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value)     \
                       && (user_op::HobDataType("mask", 0) == GetDataType<mask_dtype>::value));

REGISTER_FUSED_BIAS_ADD_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL(float, bool)
REGISTER_FUSED_BIAS_ADD_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL(half, bool)

#undef REGISTER_FUSED_BIAS_ADD_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL

}  // namespace cuda
}  // namespace oneflow
