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

namespace oneflow {

template<typename T>
T GetAttrVal(bool is_floating_val, double floating_value, int64_t integer_value) {
  return is_floating_val ? static_cast<T>(floating_value) : static_cast<T>(integer_value);
}

template<>
half GetAttrVal<half>(bool is_floating_val, double floating_value, int64_t integer_value) {
  return is_floating_val ? __float2half(floating_value) : __float2half(integer_value);
}

template<typename SRC>
struct TrilScaleMultiFetch {
  template<typename DST, int N>
  __device__ void fetch(DST* dst, int32_t row, int64_t col) {
    int32_t tril_row = row % tril_num_rows;
    bool need_fetch = (col <= (tril_row + diagonal));
    cuda::softmax::Pack<SRC, N> pack;
    if (need_fetch) {
      int64_t offset = row * row_size + col;
      pack.storage = *reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src + offset);
    }
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (col + i > tril_row + diagonal) {
        dst[i] = static_cast<DST>(fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(scale);
      }
    }
  }

  const SRC* src;
  int32_t tril_num_rows;
  int64_t row_size;
  int64_t diagonal;
  SRC fill;
  SRC scale;
};

template<typename DST>
struct MaskAndScaleMultiStore {
  template<typename SRC, int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> pack;
    int64_t offset = row * row_size + col;

    cuda::softmax::Pack<int8_t, N> mask_pack;
    mask_pack.storage = *reinterpret_cast<const cuda::softmax::PackType<int8_t, N>*>(mask + offset);

#pragma unroll
    for (int i = 0; i < N; ++i) {
      pack.elem[i] =
          static_cast<DST>(src[i]) * static_cast<DST>(mask_pack.elem[i]) * static_cast<DST>(scale);
    }
    *reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst + offset) = pack.storage;
  }
  DST* dst;
  int64_t row_size;
  const int8_t* mask;
  DST scale;
};

template<typename SRC>
struct MaskAndScaleMultiFetch {
  template<typename DST, int N>
  __device__ void fetch(DST* dst, int64_t row, int64_t col) const {
    cuda::softmax::Pack<SRC, N> pack;
    int64_t offset = row * row_size + col;
    pack.storage = *reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(mask[offset + i])
               * static_cast<DST>(scale);
    }
  }

  const SRC* src;
  int64_t row_size;
  const int8_t* mask;
  SRC scale;
};

template<typename DST>
struct TrilScaleMultiStore {
  template<typename SRC, int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> pack;
    int64_t offset = row * row_size + col;
    int64_t tril_row = row % tril_num_rows;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (col + i > tril_row + diagonal) {
        pack.elem[i] = fill;
      } else {
        pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(scale);
      }
    }
    *reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst + offset) = pack.storage;
  }
  DST* dst;
  int64_t tril_num_rows;
  int64_t row_size;
  int64_t diagonal;
  DST fill;
  DST scale;
};

template<typename T>
class FusedTrilScaleSoftmaxMaskAndScaleKernel final : public user_op::OpKernel {
 public:
  FusedTrilScaleSoftmaxMaskAndScaleKernel() = default;
  ~FusedTrilScaleSoftmaxMaskAndScaleKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape();
    const int64_t cols = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t rows = in_shape.Count(0, in_shape.NumAxes() - 1);

    const T fill = GetAttrVal<T>(ctx->Attr<bool>("is_floating_fill_value"),
                                 ctx->Attr<double>("floating_fill_value"),
                                 ctx->Attr<int64_t>("integer_fill_value"));
    const T scale = GetAttrVal<T>(ctx->Attr<bool>("is_floating_scale_value"),
                                  ctx->Attr<double>("floating_scale_value"),
                                  ctx->Attr<int64_t>("integer_scale_value"));
    TrilScaleMultiFetch<T> multi_fetch;
    multi_fetch.src = in->dptr<T>();
    multi_fetch.tril_num_rows = in_shape.At(in_shape.NumAxes() - 2);
    multi_fetch.row_size = cols;
    multi_fetch.diagonal = ctx->Attr<int64_t>("diagonal");
    multi_fetch.fill = fill;
    multi_fetch.scale = scale;
    MaskAndScaleMultiStore<T> multi_store;
    multi_store.dst = out->mut_dptr<T>();
    multi_store.mask = mask->dptr<int8_t>();
    multi_store.row_size = cols;
    multi_store.scale = ctx->Attr<float>("scale");
    cuda::softmax::DispatchSoftmax<decltype(multi_fetch), decltype(multi_store), T>(
        ctx->device_ctx()->cuda_stream(), multi_fetch, multi_store, rows, cols);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GPU_KERNEL(dtype) \
  REGISTER_USER_KERNEL("fused_tril_scale_softmax_mask_and_scale")          \
      .SetCreateFn<FusedTrilScaleSoftmaxMaskAndScaleKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)       \
                       & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GPU_KERNEL(half)
REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GPU_KERNEL(float)
REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GPU_KERNEL(double)
#undef REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GPU_KERNEL

template<typename T>
class FusedTrilScaleSoftmaxMaskAndScaleGradKernel final : public user_op::OpKernel {
 public:
  FusedTrilScaleSoftmaxMaskAndScaleGradKernel() = default;
  ~FusedTrilScaleSoftmaxMaskAndScaleGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t cols = y->shape().At(y->shape().NumAxes() - 1);
    const int64_t rows = y->shape().elem_cnt() / cols;
    const T scale = GetAttrVal<T>(ctx->Attr<bool>("is_floating_scale_value"),
                                  ctx->Attr<double>("floating_scale_value"),
                                  ctx->Attr<int64_t>("integer_scale_value"));

    cuda::softmax::MultiFetch<T> multi_fetch_y;
    multi_fetch_y.src = y->dptr<T>();
    multi_fetch_y.row_size = cols;
    cuda::softmax::MultiFetch<T> multi_fetch_dy;
    multi_fetch_dy.src = dy->dptr<T>();
    multi_fetch_dy.row_size = cols;
    TrilScaleMultiStore<T> multi_store;
    CHECK_NOTNULL(dx);
    multi_store.dst = dx->mut_dptr<T>();
    multi_store.tril_num_rows = y->shape().At(y->shape().NumAxes() - 2);
    multi_store.row_size = cols;
    multi_store.diagonal = ctx->Attr<int64_t>("diagonal");
    multi_store.fill = static_cast<T>(0);
    multi_store.scale = scale;

    cuda::softmax::DispatchSoftmaxGrad<decltype(multi_fetch_y), decltype(multi_fetch_dy),
                                       decltype(multi_store), T>(
        ctx->device_ctx()->cuda_stream(), multi_fetch_y, multi_fetch_dy, multi_store, rows, cols);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GRAD_KERNEL(dtype) \
  REGISTER_USER_KERNEL("fused_tril_scale_softmax_mask_and_scale_grad")      \
      .SetCreateFn<FusedTrilScaleSoftmaxMaskAndScaleGradKernel<dtype>>()    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU)        \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GRAD_KERNEL(half)
REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GRAD_KERNEL(float)
REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GRAD_KERNEL(double)
#undef REGISTER_FUSED_TRIL_SCALE_SOFTMAX_MASK_AND_SCALE_GRAD_KERNEL

}  // namespace oneflow
