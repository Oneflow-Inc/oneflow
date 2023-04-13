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

namespace oneflow {

template<typename SRC, typename DST>
struct TrilScaleLoad {
  TrilScaleLoad(const SRC* src, int64_t tril_num_rows, int64_t row_size, int64_t diagonal, SRC fill,
                SRC scale)
      : src(src),
        tril_num_rows(tril_num_rows),
        row_size(row_size),
        diagonal(diagonal),
        fill(fill),
        scale(scale) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) {
    int64_t tril_row = row % tril_num_rows;
    int64_t diagonal_col_id = tril_row + diagonal;
    bool need_load = (col <= diagonal_col_id);
    cuda::softmax::Pack<SRC, N> pack;
    if (need_load) {
      const int64_t offset = (row * row_size + col) / N;
      pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src) + offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (col + i > diagonal_col_id) {
        dst[i] = static_cast<DST>(fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(scale);
      }
    }
  }
  const SRC* src;
  int64_t tril_num_rows;
  int64_t row_size;
  int64_t diagonal;
  SRC fill;
  SRC scale;
};

template<typename SRC, typename DST>
struct MaskAndScaleStore {
  MaskAndScaleStore(DST* dst, DST* softmax_y, const bool* mask, int64_t row_size, DST scale)
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

template<typename SRC, typename DST>
struct MaskAndScaleLoad {
  MaskAndScaleLoad(const SRC* src, const bool* mask, int64_t row_size, SRC scale)
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
struct TrilScaleStore {
  TrilScaleStore(DST* dst, int64_t tril_num_rows, int64_t row_size, int64_t diagonal, DST fill,
                 DST scale)
      : dst(dst),
        tril_num_rows(tril_num_rows),
        row_size(row_size),
        diagonal(diagonal),
        fill(fill),
        scale(scale) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    int64_t tril_row = row % tril_num_rows;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (col + i > tril_row + diagonal) {
        pack.elem[i] = fill;
      } else {
        pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(scale);
      }
    }
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  int64_t tril_num_rows;
  int64_t row_size;
  int64_t diagonal;
  DST fill;
  DST scale;
};

template<typename T>
class FusedTrilScaleSoftmaxMaskScaleKernel final : public user_op::OpKernel {
 public:
  FusedTrilScaleSoftmaxMaskScaleKernel() = default;
  ~FusedTrilScaleSoftmaxMaskScaleKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* softmax_y = ctx->Tensor4ArgNameAndIndex("softmax_y", 0);
    const ShapeView& x_shape = x->shape_view();
    CHECK_GE(x_shape.NumAxes(), 2);
    const int64_t cols = x_shape.At(x_shape.NumAxes() - 1);
    const int64_t rows = x_shape.Count(0, x_shape.NumAxes() - 1);
    const int64_t tril_num_rows = x_shape.At(x_shape.NumAxes() - 2);
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    TrilScaleLoad<T, ComputeType> load(
        x->dptr<T>(), tril_num_rows, cols, ctx->Attr<int64_t>("diagonal"),
        ctx->Attr<float>("tril_fill_value"), ctx->Attr<float>("tril_scale_value"));
    MaskAndScaleStore<ComputeType, T> store(y->mut_dptr<T>(), softmax_y->mut_dptr<T>(),
                                            mask->dptr<bool>(), cols,
                                            ctx->Attr<float>("mask_scale_value"));
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load, store, rows, cols)));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_CUDA_KERNEL(dtype) \
  REGISTER_USER_KERNEL("fused_tril_scale_softmax_mask_scale")           \
      .SetCreateFn<FusedTrilScaleSoftmaxMaskScaleKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)  \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_CUDA_KERNEL(half)
REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_CUDA_KERNEL(float)
REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_CUDA_KERNEL(double)
#undef REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_CUDA_KERNEL

template<typename T>
class FusedTrilScaleSoftmaxMaskScaleGradKernel final : public user_op::OpKernel {
 public:
  FusedTrilScaleSoftmaxMaskScaleGradKernel() = default;
  ~FusedTrilScaleSoftmaxMaskScaleGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* softmax_y = ctx->Tensor4ArgNameAndIndex("softmax_y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ShapeView& dy_shape = dy->shape_view();
    CHECK_GE(dy_shape.NumAxes(), 2);
    const int64_t cols = dy_shape.At(dy_shape.NumAxes() - 1);
    const int64_t rows = dy_shape.Count(0, dy_shape.NumAxes() - 1);
    const int64_t tril_num_rows = dy_shape.At(dy_shape.NumAxes() - 2);
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    cuda::softmax::DirectLoad<T, ComputeType> load_softmax_y(softmax_y->dptr<T>(), cols);
    MaskAndScaleLoad<T, ComputeType> load_dy(dy->dptr<T>(), mask->dptr<bool>(), cols,
                                             ctx->Attr<float>("mask_scale_value"));
    TrilScaleStore<ComputeType, T> store(dx->mut_dptr<T>(), tril_num_rows, cols,
                                         ctx->Attr<int64_t>("diagonal"), static_cast<T>(0.0),
                                         ctx->Attr<float>("tril_scale_value"));
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_softmax_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load_softmax_y, load_dy, store, rows,
        cols)));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_GRAD_KERNEL(dtype) \
  REGISTER_USER_KERNEL("fused_tril_scale_softmax_mask_scale_grad")      \
      .SetCreateFn<FusedTrilScaleSoftmaxMaskScaleGradKernel<dtype>>()   \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)  \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_GRAD_KERNEL(half)
REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_GRAD_KERNEL(float)
REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_GRAD_KERNEL(double)
#undef REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_SCALE_GRAD_KERNEL

}  // namespace oneflow
