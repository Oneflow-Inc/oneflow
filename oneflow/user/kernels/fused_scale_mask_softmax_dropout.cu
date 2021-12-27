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
#include <cstdint>
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/cuda/softmax.cuh"
#include "oneflow/core/ep/cuda/cuda_stream.h"
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {

template<size_t num_dims, typename IndexType>
struct BroadcastMaskSoftmaxParams {
  NdIndexOffsetHelper<IndexType, num_dims> src_index_helper;
  NdIndexOffsetHelper<IndexType, num_dims> mask_index_helper;
  const int64_t* mask_dims{};
  int64_t row_size;
  float fill;
  float scale;
};

template<typename SRC, typename DST, typename MASK, size_t num_dims, typename IndexType>
struct ScaleMaskLoad {
  ScaleMaskLoad(const SRC* src, const MASK* mask,
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

template<typename SRC, typename DST, typename MASK, size_t num_dims, typename IndexType>
struct ScaleMaskStore {
  ScaleMaskStore(DST* dst, const MASK* mask, BroadcastMaskSoftmaxParams<num_dims, IndexType> params)
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

template<typename SRC, typename DST>
struct DropoutLoad {
  DropoutLoad(const SRC* src, const int8_t* mask, int64_t row_size, SRC scale)
      : src(src), mask(mask), row_size(row_size), scale(scale) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src) + offset);
    cuda::softmax::Pack<int8_t, N> mask_pack;
    mask_pack.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<int8_t, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(mask_pack.elem[i])
               * static_cast<DST>(scale);
    }
  }
  const SRC* src;
  const int8_t* mask;
  int64_t row_size;
  SRC scale;
};

template<typename SRC, typename DST>
struct DropoutStore {
  DropoutStore(DST* dst, DST* softmax_y, const int8_t* mask, int64_t row_size, DST scale)
      : dst(dst), softmax_y(softmax_y), mask(mask), row_size(row_size), scale(scale) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> softmax_y_pack;
    cuda::softmax::Pack<DST, N> dst_pack;
    const int64_t offset = (row * row_size + col) / N;
    cuda::softmax::Pack<int8_t, N> mask_pack;
    mask_pack.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<int8_t, N>*>(mask) + offset);
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
  const int8_t* mask;
  int64_t row_size;
  DST scale;
};

template<typename LOAD, typename STORE, typename ComputeType>
inline typename std::enable_if<std::is_same<ComputeType, double>::value, cudaError_t>::type
DispatchFusedSoftmax(cudaStream_t stream, LOAD load, STORE store, const int64_t rows,
                     const int64_t cols) {
  return cuda::softmax::DispatchSoftmaxBlockUncachedImpl<LOAD, STORE, ComputeType,
                                                         cuda::softmax::Algorithm::kSoftmax>(
      stream, load, store, rows, cols);
}

template<typename T, typename ComputeType, typename MASK, int num_dims>
void LaunchForwardKernel(cudaStream_t stream, const T* x, T* y, T* softmax_y, const MASK* mask,
                         const int8_t* dropout_mask, const int64_t elem_cnt, const int64_t rows,
                         const int64_t cols, const float fill, const float scale,
                         const float dropout_scale, const int64_t* input_dims,
                         const int64_t* mask_dims) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    NdIndexOffsetHelper<int32_t, num_dims> input_index_helper(input_dims, num_dims);
    NdIndexOffsetHelper<int32_t, num_dims> mask_index_helper(mask_dims, num_dims);
    BroadcastMaskSoftmaxParams<num_dims, int32_t> params;
    params.src_index_helper = input_index_helper;
    params.mask_index_helper = mask_index_helper;
    params.mask_dims = mask_dims;
    params.row_size = cols;
    params.fill = fill;
    params.scale = scale;
    ScaleMaskLoad<T, ComputeType, MASK, num_dims, int32_t> load(x, mask, params);
    DropoutStore<ComputeType, T> store(y, softmax_y, dropout_mask, cols, dropout_scale);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        stream, load, store, rows, cols)));
  } else {
    NdIndexOffsetHelper<int64_t, num_dims> input_index_helper(input_dims, num_dims);
    NdIndexOffsetHelper<int64_t, num_dims> mask_index_helper(mask_dims, num_dims);
    BroadcastMaskSoftmaxParams<num_dims, int64_t> params;
    params.src_index_helper = input_index_helper;
    params.mask_index_helper = mask_index_helper;
    params.mask_dims = mask_dims;
    params.row_size = cols;
    params.fill = fill;
    params.scale = scale;
    ScaleMaskLoad<T, ComputeType, MASK, num_dims, int64_t> load(x, mask, params);
    DropoutStore<ComputeType, T> store(y, softmax_y, dropout_mask, cols, dropout_scale);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        stream, load, store, rows, cols)));
  }
}

template<typename T, typename ComputeType, typename MASK, int num_dims>
void LaunchBackwardKernel(cudaStream_t stream, const T* softmax_y, const T* dy, T* dx,
                          const MASK* mask, const int8_t* dropout_mask, const int64_t elem_cnt,
                          const int64_t rows, const int64_t cols, const float fill,
                          const float scale, const float dropout_scale, const int64_t* input_dims,
                          const int64_t* mask_dims) {
  if (elem_cnt < GetMaxVal<int32_t>()) {
    NdIndexOffsetHelper<int32_t, num_dims> input_index_helper(input_dims, num_dims);
    NdIndexOffsetHelper<int32_t, num_dims> mask_index_helper(mask_dims, num_dims);
    BroadcastMaskSoftmaxParams<num_dims, int32_t> params;
    params.src_index_helper = input_index_helper;
    params.mask_index_helper = mask_index_helper;
    params.mask_dims = mask_dims;
    params.row_size = cols;
    params.fill = fill;
    params.scale = scale;
    cuda::softmax::DirectLoad<T, ComputeType> load_softmax_y(softmax_y, cols);
    DropoutLoad<T, ComputeType> load_dy(dy, dropout_mask, cols, dropout_scale);
    ScaleMaskStore<ComputeType, T, MASK, num_dims, int32_t> store(dx, mask, params);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_softmax_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
        stream, load_softmax_y, load_dy, store, rows, cols)));
  } else {
    NdIndexOffsetHelper<int64_t, num_dims> input_index_helper(input_dims, num_dims);
    NdIndexOffsetHelper<int64_t, num_dims> mask_index_helper(mask_dims, num_dims);
    BroadcastMaskSoftmaxParams<num_dims, int64_t> params;
    params.src_index_helper = input_index_helper;
    params.mask_index_helper = mask_index_helper;
    params.mask_dims = mask_dims;
    params.row_size = cols;
    params.fill = fill;
    params.scale = scale;
    cuda::softmax::DirectLoad<T, ComputeType> load_softmax_y(softmax_y, cols);
    DropoutLoad<T, ComputeType> load_dy(dy, dropout_mask, cols, dropout_scale);
    ScaleMaskStore<ComputeType, T, MASK, num_dims, int64_t> store(dx, mask, params);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_softmax_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
        stream, load_softmax_y, load_dy, store, rows, cols)));
  }
}

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

constexpr int32_t kMaxNumDims = 8;

template<typename T, typename MASK>
class FusedScaleMaskSoftmaxDropoutKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskSoftmaxDropoutKernel() = default;
  ~FusedScaleMaskSoftmaxDropoutKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const user_op::Tensor* dropout_mask = ctx->Tensor4ArgNameAndIndex("dropout_mask", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* softmax_y = ctx->Tensor4ArgNameAndIndex("softmax_y", 0);
    const ShapeView& x_shape = x->shape();
    const ShapeView& mask_shape = mask->shape();
    CHECK_GE(x_shape.NumAxes(), 2);
    const int64_t elem_cnt = x_shape.elem_cnt();
    const int64_t cols = x_shape.At(x_shape.NumAxes() - 1);
    const int64_t rows = x_shape.Count(0, x_shape.NumAxes() - 1);
    const size_t num_input_dims = x_shape.NumAxes();
    const int64_t* input_dims = x_shape.ptr();
    const size_t num_mask_dims = mask_shape.NumAxes();
    const int64_t* mask_dims = mask_shape.ptr();
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;

    size_t simplified_num_dims = 0;
    int64_t simplified_input_dims[kMaxNumDims];
    int64_t simplified_mask_dims[kMaxNumDims];
    SimplifyBroadcastDims(num_input_dims, input_dims, num_mask_dims, mask_dims,
                          &simplified_num_dims, simplified_input_dims, simplified_mask_dims);

    printf("Here simplified num dims is: %d \n", simplified_num_dims);

    if (simplified_num_dims == 2) {
      LaunchForwardKernel<T, ComputeType, MASK, 2>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), x->dptr<T>(), y->mut_dptr<T>(),
          softmax_y->mut_dptr<T>(), mask->dptr<MASK>(), dropout_mask->dptr<int8_t>(), elem_cnt,
          rows, cols, ctx->Attr<float>("mask_fill_value"), ctx->Attr<float>("scale_value"),
          ctx->Attr<float>("dropout_scale_value"), simplified_input_dims, simplified_mask_dims);
    }
#define DEFINE_ONE_ELIF(dims)                                                                 \
  else if (simplified_num_dims == dims) {                                                     \
    LaunchForwardKernel<T, ComputeType, MASK, dims>(                                          \
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), x->dptr<T>(), y->mut_dptr<T>(),   \
        softmax_y->mut_dptr<T>(), mask->dptr<MASK>(), dropout_mask->dptr<int8_t>(), elem_cnt, \
        rows, cols, ctx->Attr<float>("mask_fill_value"), ctx->Attr<float>("scale_value"),     \
        ctx->Attr<float>("dropout_scale_value"), input_dims, mask_dims);                      \
  }
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(5)
    DEFINE_ONE_ELIF(6)
#undef DEFINE_ONE_ELIF
    else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T, typename MASK>
class FusedScaleMaskSoftmaxDropoutGradKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskSoftmaxDropoutGradKernel() = default;
  ~FusedScaleMaskSoftmaxDropoutGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* softmax_y = ctx->Tensor4ArgNameAndIndex("softmax_y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    const user_op::Tensor* dropout_mask = ctx->Tensor4ArgNameAndIndex("dropout_mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ShapeView& dy_shape = dy->shape();
    const int64_t elem_cnt = dy_shape.elem_cnt();
    const ShapeView& mask_shape = mask->shape();
    CHECK_GE(dy_shape.NumAxes(), 2);
    const int64_t cols = dy_shape.At(dy_shape.NumAxes() - 1);
    const int64_t rows = dy_shape.Count(0, dy_shape.NumAxes() - 1);
    const int64_t* input_dims = dy_shape.ptr();
    const size_t num_input_dims = dy_shape.NumAxes();
    const int64_t* mask_dims = mask_shape.ptr();
    const size_t num_mask_dims = mask_shape.NumAxes();

    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    cuda::softmax::DirectLoad<T, ComputeType> load_softmax_y(softmax_y->dptr<T>(), cols);

    size_t simplified_num_dims = 0;
    int64_t simplified_input_dims[kMaxNumDims];
    int64_t simplified_mask_dims[kMaxNumDims];
    SimplifyBroadcastDims(num_input_dims, input_dims, num_mask_dims, mask_dims,
                          &simplified_num_dims, simplified_input_dims, simplified_mask_dims);

    printf("Here simplified num dims is: %d \n", simplified_num_dims);

    if (simplified_num_dims == 2) {
      LaunchBackwardKernel<T, ComputeType, MASK, 2>(
          ctx->stream()->As<ep::CudaStream>()->cuda_stream(), softmax_y->dptr<T>(), dy->dptr<T>(),
          dx->mut_dptr<T>(), mask->dptr<MASK>(), dropout_mask->dptr<int8_t>(), elem_cnt, rows, cols,
          static_cast<float>(0.0), ctx->Attr<float>("scale_value"),
          ctx->Attr<float>("dropout_scale_value"), simplified_input_dims, simplified_mask_dims);
    }
#define DEFINE_ONE_ELIF(dims)                                                                      \
  else if (simplified_num_dims == dims) {                                                          \
    LaunchBackwardKernel<T, ComputeType, MASK, dims>(                                              \
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), softmax_y->dptr<T>(), dy->dptr<T>(),   \
        dx->mut_dptr<T>(), mask->dptr<MASK>(), dropout_mask->dptr<int8_t>(), elem_cnt, rows, cols, \
        static_cast<float>(0.0), ctx->Attr<float>("scale_value"),                                  \
        ctx->Attr<float>("dropout_scale_value"), simplified_input_dims, simplified_mask_dims);     \
  }
    DEFINE_ONE_ELIF(3)
    DEFINE_ONE_ELIF(4)
    DEFINE_ONE_ELIF(5)
    DEFINE_ONE_ELIF(6)
#undef DEFINE_ONE_ELIF
    else {
      UNIMPLEMENTED();
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

}  // namespace

#define REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL(dtype, mask_dtype)      \
  REGISTER_USER_KERNEL("fused_scale_mask_softmax_dropout")                            \
      .SetCreateFn<FusedScaleMaskSoftmaxDropoutKernel<dtype, mask_dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                \
                       && (user_op::HobDataType("x", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("mask", 0) == GetDataType<mask_dtype>::value));

// REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL(half, int8_t)
REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL(float, int8_t)
// REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL(double, int8_t)
#undef REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_CUDA_KERNEL

#define REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_GRAD_KERNEL(dtype, mask_dtype)       \
  REGISTER_USER_KERNEL("fused_scale_mask_softmax_dropout_grad")                        \
      .SetCreateFn<FusedScaleMaskSoftmaxDropoutGradKernel<dtype, mask_dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)                 \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value) \
                       && (user_op::HobDataType("mask", 0) == GetDataType<mask_dtype>::value));

// REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_GRAD_KERNEL(half, int8_t)
REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_GRAD_KERNEL(float, int8_t)
// REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_GRAD_KERNEL(double, int8_t)
#undef REGISTER_FUSED_SCALE_MASK_SOFTMAX_DROPOUT_GRAD_KERNEL

}  // namespace oneflow
