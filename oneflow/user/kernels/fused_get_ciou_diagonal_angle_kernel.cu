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
#include <cmath>
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<typename T>
struct FusedCiouAngleForwardFunctor {
  __device__ T Compute(T w1, T h1, T w2, T h2, T eps) const {
    T angle = (atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps)))
              * (atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps)));
    return static_cast<T>(4.0 / (M_PI * M_PI)) * angle;
  }
};

template<>
struct FusedCiouAngleForwardFunctor<half> {
  __device__ half Compute(half w1, half h1, half w2, half h2, float eps) const {
    float w1f = __half2float(w1);
    float h1f = __half2float(h1);
    float w2f = __half2float(w2);
    float h2f = __half2float(h2);
    float angle = (atan(w2f / (h2f + eps)) - atan(w1f / (h1f + eps)))
                  * (atan(w2f / (h2f + eps)) - atan(w1f / (h1f + eps)));
    return static_cast<half>(4.0 / (M_PI * M_PI)) * __float2half(angle);
  }
};

template<typename FUNCTOR, typename T>
__global__ void FusedCiouAngleForward(FUNCTOR functor, const int n, const T* w1, const T* h1,
                                      const T* w2, const T* h2, const T eps, T* v) {
  CUDA_1D_KERNEL_LOOP(i, n) { v[i] = functor.Compute(w1[i], h1[i], w2[i], h2[i], eps); }
}

template<typename T>
struct FusedCiouAngleBackwardFunctor {
  __device__ T ComputeW1(T w1, T h1, T w2, T h2, T eps) const {
    T angle_delta = atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps));
    return static_cast<T>(-8) * angle_delta
           / (static_cast<T>((M_PI * M_PI)) * (h1 + eps)
              * (static_cast<T>(1) + (w1 * w1 / ((h1 + eps) * (h1 + eps)))));
  }

  __device__ T ComputeW2(T w1, T h1, T w2, T h2, T eps) const {
    T angle_delta = atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps));
    return static_cast<T>(8) * angle_delta
           / (static_cast<T>((M_PI * M_PI)) * (h2 + eps)
              * (static_cast<T>(1) + (w2 * w2 / ((h2 + eps) * (h2 + eps)))));
  }

  __device__ T ComputeH1(T w1, T h1, T w2, T h2, T eps) const {
    T angle_delta = atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps));
    return static_cast<T>(8) * w1 * angle_delta
           / (static_cast<T>((M_PI * M_PI)) * (h1 + eps) * (h1 + eps)
              * (static_cast<T>(1) + (w1 * w1 / ((h1 + eps) * (h1 + eps)))));
  }

  __device__ T ComputeH2(T w1, T h1, T w2, T h2, T eps) const {
    T angle_delta = atan(w2 / (h2 + eps)) - atan(w1 / (h1 + eps));
    return static_cast<T>(-8) * w2 * angle_delta
           / (static_cast<T>(M_PI * M_PI) * (h2 + eps) * (h2 + eps)
              * (static_cast<T>(1) + (w2 * w2 / ((h2 + eps) * (h2 + eps)))));
  }
};

template<>
struct FusedCiouAngleBackwardFunctor<half> {
  __device__ half ComputeW1(half w1, half h1, half w2, half h2, float eps) const {
    float w1f = __half2float(w1);
    float h1f = __half2float(h1);
    float w2f = __half2float(w2);
    float h2f = __half2float(h2);
    float angle_delta = atan(w2f / (h2f + eps)) - atan(w1f / (h1f + eps));
    return __float2half(static_cast<float>(-8) * angle_delta
                        / (M_PI * M_PI * (h1f + eps)
                           * (static_cast<float>(1) + (w1f * w1f / ((h1f + eps) * (h1f + eps))))));
  }

  __device__ half ComputeW2(half w1, half h1, half w2, half h2, float eps) const {
    float w1f = __half2float(w1);
    float h1f = __half2float(h1);
    float w2f = __half2float(w2);
    float h2f = __half2float(h2);
    float angle_delta = atan(w2f / (h2f + eps)) - atan(w1f / (h1f + eps));
    return __float2half(static_cast<float>(8) * angle_delta
                        / (M_PI * M_PI * (h2f + eps)
                           * (static_cast<float>(1) + (w2f * w2f / ((h2f + eps) * (h2f + eps))))));
  }

  __device__ half ComputeH1(half w1, half h1, half w2, half h2, float eps) const {
    float w1f = __half2float(w1);
    float h1f = __half2float(h1);
    float w2f = __half2float(w2);
    float h2f = __half2float(h2);
    float angle_delta = atan(w2f / (h2f + eps)) - atan(w1f / (h1f + eps));
    return __float2half(static_cast<float>(8) * w1f * angle_delta
                        / (M_PI * M_PI * (h1f + eps) * (h1f + eps)
                           * (static_cast<float>(1) + (w1f * w1f / ((h1f + eps) * (h1f + eps))))));
  }

  __device__ half ComputeH2(half w1, half h1, half w2, half h2, float eps) const {
    float w1f = __half2float(w1);
    float h1f = __half2float(h1);
    float w2f = __half2float(w2);
    float h2f = __half2float(h2);
    float angle_delta = atan(w2f / (h2f + eps)) - atan(w1f / (h1f + eps));
    return __float2half(static_cast<float>(-8) * w2f * angle_delta
                        / (M_PI * M_PI * (h2f + eps) * (h2f + eps)
                           * (static_cast<float>(1) + (w2f * w2f / ((h2f + eps) * (h2f + eps))))));
  }
};

template<typename FUNCTOR, typename T>
__global__ void FusedCiouAngleBackward(FUNCTOR functor, const int n, const T* w1, const T* h1,
                                       const T* w2, const T* h2, const T* v_diff, const T eps,
                                       T* w1_diff, T* h1_diff, T* w2_diff, T* h2_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    w1_diff[i] = functor.ComputeW1(w1[i], h1[i], w2[i], h2[i], eps) * v_diff[i];
    w2_diff[i] = functor.ComputeW2(w1[i], h1[i], w2[i], h2[i], eps) * v_diff[i];
    h1_diff[i] = functor.ComputeH1(w1[i], h1[i], w2[i], h2[i], eps) * v_diff[i];
    h2_diff[i] = functor.ComputeH2(w1[i], h1[i], w2[i], h2[i], eps) * v_diff[i];
  }
}

}  // namespace

template<typename T>
class FusedGetCiouDiagonalAngleKernel final : public user_op::OpKernel {
 public:
  FusedGetCiouDiagonalAngleKernel() = default;
  ~FusedGetCiouDiagonalAngleKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* w1 = ctx->Tensor4ArgNameAndIndex("w1", 0);
    const user_op::Tensor* h1 = ctx->Tensor4ArgNameAndIndex("h1", 0);
    const user_op::Tensor* w2 = ctx->Tensor4ArgNameAndIndex("w2", 0);
    const user_op::Tensor* h2 = ctx->Tensor4ArgNameAndIndex("h2", 0);
    const auto eps = ctx->Attr<float>("eps");

    user_op::Tensor* v = ctx->Tensor4ArgNameAndIndex("v", 0);

    const int64_t elem_cnt = w1->shape_view().elem_cnt();

    FusedCiouAngleForwardFunctor<T> fused_get_ciou_diagonal_angle_functor{};

    RUN_CUDA_KERNEL((FusedCiouAngleForward<decltype(fused_get_ciou_diagonal_angle_functor), T>),
                    ctx->stream(), elem_cnt, fused_get_ciou_diagonal_angle_functor, elem_cnt,
                    w1->dptr<T>(), h1->dptr<T>(), w2->dptr<T>(), h2->dptr<T>(), static_cast<T>(eps),
                    v->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_CIOU_DIAGONAL_ANGLE_CUDA_KERNEL(dtype)      \
  REGISTER_USER_KERNEL("fused_get_ciou_diagonal_angle")                \
      .SetCreateFn<FusedGetCiouDiagonalAngleKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("v", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_CIOU_DIAGONAL_ANGLE_CUDA_KERNEL(float)
REGISTER_FUSED_GET_CIOU_DIAGONAL_ANGLE_CUDA_KERNEL(double)
REGISTER_FUSED_GET_CIOU_DIAGONAL_ANGLE_CUDA_KERNEL(half)

template<typename T>
class FusedGetCiouDiagonalAngleGradKernel final : public user_op::OpKernel {
 public:
  FusedGetCiouDiagonalAngleGradKernel() = default;
  ~FusedGetCiouDiagonalAngleGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* w1 = ctx->Tensor4ArgNameAndIndex("w1", 0);
    const user_op::Tensor* h1 = ctx->Tensor4ArgNameAndIndex("h1", 0);
    const user_op::Tensor* w2 = ctx->Tensor4ArgNameAndIndex("w2", 0);
    const user_op::Tensor* h2 = ctx->Tensor4ArgNameAndIndex("h2", 0);
    const user_op::Tensor* v_diff = ctx->Tensor4ArgNameAndIndex("v_diff", 0);
    const auto eps = ctx->Attr<float>("eps");

    user_op::Tensor* w1_diff = ctx->Tensor4ArgNameAndIndex("w1_diff", 0);
    user_op::Tensor* h1_diff = ctx->Tensor4ArgNameAndIndex("h1_diff", 0);
    user_op::Tensor* w2_diff = ctx->Tensor4ArgNameAndIndex("w2_diff", 0);
    user_op::Tensor* h2_diff = ctx->Tensor4ArgNameAndIndex("h2_diff", 0);

    const int64_t elem_cnt = w1->shape_view().elem_cnt();

    FusedCiouAngleBackwardFunctor<T> fused_get_ciou_diagonal_angle_grad_functor{};

    RUN_CUDA_KERNEL(
        (FusedCiouAngleBackward<decltype(fused_get_ciou_diagonal_angle_grad_functor), T>),
        ctx->stream(), elem_cnt, fused_get_ciou_diagonal_angle_grad_functor, elem_cnt,
        w1->dptr<T>(), h1->dptr<T>(), w2->dptr<T>(), h2->dptr<T>(), v_diff->dptr<T>(),
        static_cast<T>(eps), w1_diff->mut_dptr<T>(), h1_diff->mut_dptr<T>(), w2_diff->mut_dptr<T>(),
        h2_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_CIOU_DIAGONAL_ANGLE_GRAD_CUDA_KERNEL(dtype) \
  REGISTER_USER_KERNEL("fused_get_ciou_diagonal_angle_grad")           \
      .SetCreateFn<FusedGetCiouDiagonalAngleGradKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("w1_diff", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_CIOU_DIAGONAL_ANGLE_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_GET_CIOU_DIAGONAL_ANGLE_GRAD_CUDA_KERNEL(double)
REGISTER_FUSED_GET_CIOU_DIAGONAL_ANGLE_GRAD_CUDA_KERNEL(half)

}  // namespace oneflow
