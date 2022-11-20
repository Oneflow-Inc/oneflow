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
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
struct FusedCenterForwardFunctor {
  __device__ T Compute(T b1_x1, T b1_x2, T b2_x1, T b2_x2, T b1_y1, T b1_y2, T b2_y1,
                       T b2_y2) const {
    return ((b2_x1 + b2_x2 - b1_x1 - b1_x2) * (b2_x1 + b2_x2 - b1_x1 - b1_x2)
            + (b2_y1 + b2_y2 - b1_y1 - b1_y2) * (b2_y1 + b2_y2 - b1_y1 - b1_y2))
           / static_cast<T>(4.0);
  }
};

template<>
struct FusedCenterForwardFunctor<half> {
  FusedCenterForwardFunctor<float> float_functor;
  __device__ half Compute(half b1_x1, half b1_x2, half b2_x1, half b2_x2, half b1_y1, half b1_y2,
                          half b2_y1, half b2_y2) const {
    return __float2half(float_functor.Compute(
        __half2float(b1_x1), __half2float(b1_x2), __half2float(b2_x1), __half2float(b2_x2),
        __half2float(b1_y1), __half2float(b1_y2), __half2float(b2_y1), __half2float(b2_y2)));
  }
};

template<typename FUNCTOR, typename T>
__global__ void FusedCenterForward(FUNCTOR functor, const int n, const T* b1_x1, const T* b1_x2,
                                   const T* b2_x1, const T* b2_x2, const T* b1_y1, const T* b1_y2,
                                   const T* b2_y1, const T* b2_y2, T* rho) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    rho[i] = functor.Compute(b1_x1[i], b1_x2[i], b2_x1[i], b2_x2[i], b1_y1[i], b1_y2[i], b2_y1[i],
                             b2_y2[i]);
  }
}

template<typename T>
__global__ void FusedCenterBackward(const int n, const T* b1_x1, const T* b1_x2, const T* b2_x1,
                                    const T* b2_x2, const T* b1_y1, const T* b1_y2, const T* b2_y1,
                                    const T* b2_y2, T* b1_x1_diff, T* b1_x2_diff, T* b2_x1_diff,
                                    T* b2_x2_diff, T* b1_y1_diff, T* b1_y2_diff, T* b2_y1_diff,
                                    T* b2_y2_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T b_x_diff = b1_x1[i] + b1_x2[i] - b2_x1[i] - b2_x2[i];
    const T b_y_diff = b1_y1[i] + b1_y2[i] - b2_y1[i] - b2_y2[i];

    b1_x1_diff[i] = b_x_diff / static_cast<T>(2.0);
    b1_x2_diff[i] = b_x_diff / static_cast<T>(2.0);
    b2_x1_diff[i] = b_x_diff / static_cast<T>(-2.0);
    b2_x2_diff[i] = b_x_diff / static_cast<T>(-2.0);

    b1_y1_diff[i] = b_y_diff / static_cast<T>(2.0);
    b1_y2_diff[i] = b_y_diff / static_cast<T>(2.0);
    b2_y1_diff[i] = b_y_diff / static_cast<T>(-2.0);
    b2_y2_diff[i] = b_y_diff / static_cast<T>(-2.0);
  }
}

}  // namespace

template<typename T>
class FusedCenterKernel final : public user_op::OpKernel {
 public:
  FusedCenterKernel() = default;
  ~FusedCenterKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* b1_x1 = ctx->Tensor4ArgNameAndIndex("b1_x1", 0);
    const user_op::Tensor* b1_x2 = ctx->Tensor4ArgNameAndIndex("b1_x2", 0);
    const user_op::Tensor* b2_x1 = ctx->Tensor4ArgNameAndIndex("b2_x1", 0);
    const user_op::Tensor* b2_x2 = ctx->Tensor4ArgNameAndIndex("b2_x2", 0);
    const user_op::Tensor* b1_y1 = ctx->Tensor4ArgNameAndIndex("b1_y1", 0);
    const user_op::Tensor* b1_y2 = ctx->Tensor4ArgNameAndIndex("b1_y2", 0);
    const user_op::Tensor* b2_y1 = ctx->Tensor4ArgNameAndIndex("b2_y1", 0);
    const user_op::Tensor* b2_y2 = ctx->Tensor4ArgNameAndIndex("b2_y2", 0);

    user_op::Tensor* rho = ctx->Tensor4ArgNameAndIndex("rho2", 0);

    const int64_t elem_cnt = b1_x1->shape_view().elem_cnt();

    FusedCenterForwardFunctor<T> fused_center_forward_functor{};

    RUN_CUDA_KERNEL((FusedCenterForward<decltype(fused_center_forward_functor), T>), ctx->stream(),
                    elem_cnt, fused_center_forward_functor, elem_cnt, b1_x1->dptr<T>(),
                    b1_x2->dptr<T>(), b2_x1->dptr<T>(), b2_x2->dptr<T>(), b1_y1->dptr<T>(),
                    b1_y2->dptr<T>(), b2_y1->dptr<T>(), b2_y2->dptr<T>(), rho->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_CENTER_DIST_CUDA_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("fused_get_center_dist")                        \
      .SetCreateFn<FusedCenterKernel<dtype>>()                         \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("rho2", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_CENTER_DIST_CUDA_KERNEL(float)
REGISTER_FUSED_GET_CENTER_DIST_CUDA_KERNEL(double)
REGISTER_FUSED_GET_CENTER_DIST_CUDA_KERNEL(half)

template<typename T>
class FusedCenterGradKernel final : public user_op::OpKernel {
 public:
  FusedCenterGradKernel() = default;
  ~FusedCenterGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* b1_x1 = ctx->Tensor4ArgNameAndIndex("b1_x1", 0);
    const user_op::Tensor* b1_x2 = ctx->Tensor4ArgNameAndIndex("b1_x2", 0);
    const user_op::Tensor* b2_x1 = ctx->Tensor4ArgNameAndIndex("b2_x1", 0);
    const user_op::Tensor* b2_x2 = ctx->Tensor4ArgNameAndIndex("b2_x2", 0);
    const user_op::Tensor* b1_y1 = ctx->Tensor4ArgNameAndIndex("b1_y1", 0);
    const user_op::Tensor* b1_y2 = ctx->Tensor4ArgNameAndIndex("b1_y2", 0);
    const user_op::Tensor* b2_y1 = ctx->Tensor4ArgNameAndIndex("b2_y1", 0);
    const user_op::Tensor* b2_y2 = ctx->Tensor4ArgNameAndIndex("b2_y2", 0);

    user_op::Tensor* b1_x1_diff = ctx->Tensor4ArgNameAndIndex("b1_x1_diff", 0);
    user_op::Tensor* b1_x2_diff = ctx->Tensor4ArgNameAndIndex("b1_x2_diff", 0);
    user_op::Tensor* b2_x1_diff = ctx->Tensor4ArgNameAndIndex("b2_x1_diff", 0);
    user_op::Tensor* b2_x2_diff = ctx->Tensor4ArgNameAndIndex("b2_x2_diff", 0);
    user_op::Tensor* b1_y1_diff = ctx->Tensor4ArgNameAndIndex("b1_y1_diff", 0);
    user_op::Tensor* b1_y2_diff = ctx->Tensor4ArgNameAndIndex("b1_y2_diff", 0);
    user_op::Tensor* b2_y1_diff = ctx->Tensor4ArgNameAndIndex("b2_y1_diff", 0);
    user_op::Tensor* b2_y2_diff = ctx->Tensor4ArgNameAndIndex("b2_y2_diff", 0);

    const int64_t elem_cnt = b1_x1_diff->shape_view().elem_cnt();

    RUN_CUDA_KERNEL((FusedCenterBackward<T>), ctx->stream(), elem_cnt, elem_cnt, b1_x1->dptr<T>(),
                    b1_x2->dptr<T>(), b2_x1->dptr<T>(), b2_x2->dptr<T>(), b1_y1->dptr<T>(),
                    b1_y2->dptr<T>(), b2_y1->dptr<T>(), b2_y2->dptr<T>(), b1_x1_diff->mut_dptr<T>(),
                    b1_x2_diff->mut_dptr<T>(), b2_x1_diff->mut_dptr<T>(), b2_x2_diff->mut_dptr<T>(),
                    b1_y1_diff->mut_dptr<T>(), b1_y2_diff->mut_dptr<T>(), b2_y1_diff->mut_dptr<T>(),
                    b2_y2_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_CENTER_DIST_GRAD_CUDA_KERNEL(dtype)         \
  REGISTER_USER_KERNEL("fused_get_center_dist_grad")                   \
      .SetCreateFn<FusedCenterGradKernel<dtype>>()                     \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("b1_x1", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_CENTER_DIST_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_GET_CENTER_DIST_GRAD_CUDA_KERNEL(double)
REGISTER_FUSED_GET_CENTER_DIST_GRAD_CUDA_KERNEL(half)

}  // namespace oneflow
