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
class SoftmaxKernel final : public user_op::OpKernel {
 public:
  SoftmaxKernel() = default;
  ~SoftmaxKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const ShapeView& in_shape = in->shape();
    const int64_t cols = in_shape.At(in_shape.NumAxes() - 1);
    const int64_t rows = in_shape.Count(0, in_shape.NumAxes() - 1);
    cuda::softmax::DirectFetch<T> fetch(in->dptr<T>(), cols);
    cuda::softmax::DirectStore<T> store(out->mut_dptr<T>(), cols);
    cuda::softmax::DispatchSoftmax<decltype(fetch), decltype(store), T>(
        ctx->device_ctx()->cuda_stream(), fetch, store, rows, cols);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GPU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("softmax").SetCreateFn<SoftmaxKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == DeviceType::kGPU)                                    \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_SOFTMAX_GPU_KERNEL(half)
REGISTER_SOFTMAX_GPU_KERNEL(float)
REGISTER_SOFTMAX_GPU_KERNEL(double)
#undef REGISTER_SOFTMAX_GPU_KERNEL

template<typename T>
class SoftmaxGradKernel final : public user_op::OpKernel {
 public:
  SoftmaxGradKernel() = default;
  ~SoftmaxGradKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const int64_t cols = y->shape().At(y->shape().NumAxes() - 1);
    const int64_t rows = y->shape().elem_cnt() / cols;
    cuda::softmax::DirectFetch<T> fetch_y(y->dptr<T>(), cols);
    cuda::softmax::DirectFetch<T> fetch_dy(dy->dptr<T>(), cols);
    cuda::softmax::DirectStore<T> store(dx->mut_dptr<T>(), cols);
    cuda::softmax::DispatchSoftmaxGrad<decltype(fetch_y), decltype(fetch_dy), decltype(store), T>(
        ctx->device_ctx()->cuda_stream(), fetch_y, fetch_dy, store, rows, cols);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SOFTMAX_GRAD_KERNEL(dtype)                          \
  REGISTER_USER_KERNEL("softmax_grad")                               \
      .SetCreateFn<SoftmaxGradKernel<dtype>>()                       \
      .SetIsMatchedHob((user_op::HobDeviceTag() == DeviceType::kGPU) \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_SOFTMAX_GRAD_KERNEL(half)
REGISTER_SOFTMAX_GRAD_KERNEL(float)
REGISTER_SOFTMAX_GRAD_KERNEL(double)
#undef REGISTER_SOFTMAX_GRAD_KERNEL

}  // namespace oneflow
