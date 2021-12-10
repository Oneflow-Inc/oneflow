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
struct ScaleMaskLoad {
  ScaleMaskLoad(const SRC* src, const int8_t* mask, int64_t row_size, SRC fill, SRC scale)
      : src(src), mask(mask), row_size(row_size), fill(fill), scale(scale) {}
  template<int N>
  __device__ void load(DST* dst, int64_t row, int64_t col) {
    cuda::softmax::Pack<SRC, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    pack.storage = *(reinterpret_cast<const cuda::softmax::PackType<SRC, N>*>(src) + offset);
    cuda::softmax::Pack<int8_t, N> mask_pack;
    mask_pack.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<int8_t, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        dst[i] = static_cast<DST>(fill);
      } else {
        dst[i] = static_cast<DST>(pack.elem[i]) * static_cast<DST>(scale);
      }
    }
  }
  const SRC* src;
  const int8_t* mask;
  int64_t row_size;
  SRC fill;
  SRC scale;
};

template<typename SRC, typename DST>
struct ScaleMaskStore {
  ScaleMaskStore(DST* dst, const int8_t* mask, int64_t row_size, DST fill, DST scale)
      : dst(dst), mask(mask), row_size(row_size), fill(fill), scale(scale) {}
  template<int N>
  __device__ void store(const SRC* src, int64_t row, int64_t col) {
    cuda::softmax::Pack<DST, N> pack;
    const int64_t offset = (row * row_size + col) / N;
    cuda::softmax::Pack<int8_t, N> mask_pack;
    mask_pack.storage =
        *(reinterpret_cast<const cuda::softmax::PackType<int8_t, N>*>(mask) + offset);
#pragma unroll
    for (int i = 0; i < N; ++i) {
      if (mask_pack.elem[i] == 0) {
        pack.elem[i] = fill;
      } else {
        pack.elem[i] = static_cast<DST>(src[i]) * static_cast<DST>(scale);
      }
    }
    *(reinterpret_cast<cuda::softmax::PackType<DST, N>*>(dst) + offset) = pack.storage;
  }
  DST* dst;
  const int8_t* mask;
  int64_t row_size;
  DST fill;
  DST scale;
};

template<typename T>
class FusedScaleMaskSoftmaxKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskSoftmaxKernel() = default;
  ~FusedScaleMaskSoftmaxKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const ShapeView& x_shape = x->shape();
    CHECK_GE(x_shape.NumAxes(), 2);
    const int64_t cols = x_shape.At(x_shape.NumAxes() - 1);
    const int64_t rows = x_shape.Count(0, x_shape.NumAxes() - 1);
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    ScaleMaskLoad<T, ComputeType> load(x->dptr<T>(), mask->dptr<int8_t>(), cols,
                                       ctx->Attr<float>("mask_fill_value"),
                                       ctx->Attr<float>("scale_value"));
    cuda::softmax::DirectStore<ComputeType, T> store(y->mut_dptr<T>(), cols);
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmax<decltype(load), decltype(store), ComputeType>(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load, store, rows, cols)));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("fused_scale_mask_softmax")                     \
      .SetCreateFn<FusedScaleMaskSoftmaxKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(half)
REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(float)
REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL(double)
#undef REGISTER_FUCED_SCALE_MASK_SOFTMAX_CUDA_KERNEL

template<typename T>
class FusedScaleMaskSoftmaxGradKernel final : public user_op::OpKernel {
 public:
  FusedScaleMaskSoftmaxGradKernel() = default;
  ~FusedScaleMaskSoftmaxGradKernel() override = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* mask = ctx->Tensor4ArgNameAndIndex("mask", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const ShapeView& dy_shape = dy->shape();
    CHECK_GE(dy_shape.NumAxes(), 2);
    const int64_t cols = dy_shape.At(dy_shape.NumAxes() - 1);
    const int64_t rows = dy_shape.Count(0, dy_shape.NumAxes() - 1);
    using ComputeType = typename cuda::softmax::DefaultComputeType<T>::type;
    cuda::softmax::DirectLoad<T, ComputeType> load_y(y->dptr<T>(), cols);
    cuda::softmax::DirectLoad<T, ComputeType> load_dy(dy->dptr<T>(), cols);
    ScaleMaskStore<ComputeType, T> store(dx->mut_dptr<T>(), mask->dptr<int8_t>(), cols,
                                         static_cast<T>(0.0), ctx->Attr<float>("scale_value"));
    OF_CUDA_CHECK((cuda::softmax::DispatchSoftmaxGrad<decltype(load_y), decltype(load_dy),
                                                      decltype(store), ComputeType>(
        ctx->stream()->As<ep::CudaStream>()->cuda_stream(), load_y, load_dy, store, rows, cols)));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("fused_scale_mask_softmax_grad")                \
      .SetCreateFn<FusedScaleMaskSoftmaxGradKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(half)
REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(float)
REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL(double)
#undef REGISTER_FUCED_SCALE_MASK_SOFTMAX_GRAD_KERNEL

}  // namespace oneflow
