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
struct MinMaxDeltaFunctor {
  __device__ T Compute(T b1_x2_i, T b2_x2_i, T b1_x1_i, T b2_x1_i) const {
    return min(b1_x2_i, b2_x2_i) - max(b1_x1_i, b2_x1_i);
  }
};

template<>
struct MinMaxDeltaFunctor<half> {
  __device__ half Compute(half b1_x2_i, half b2_x2_i, half b1_x1_i, half b2_x1_i) const {
    const half b_x2_min = b1_x2_i < b2_x2_i ? b1_x2_i : b2_x2_i;
    const half b_x1_max = b1_x1_i > b2_x1_i ? b1_x1_i : b2_x1_i;
    return b_x2_min - b_x1_max;
  }
};

template<typename FUNCTOR, typename T>
__global__ void FusedGetIntersectionAreaBackward(FUNCTOR functor, const int n, const T* b1_x1,
                                                 const T* b1_x2, const T* b2_x1, const T* b2_x2,
                                                 const T* b1_y1, const T* b1_y2, const T* b2_y1,
                                                 const T* b2_y2, const T* inter_diff, T* b1_x1_diff,
                                                 T* b1_x2_diff, T* b2_x1_diff, T* b2_x2_diff,
                                                 T* b1_y1_diff, T* b1_y2_diff, T* b2_y1_diff,
                                                 T* b2_y2_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T inter_diff_i = inter_diff[i];
    const T b_x_min_max = functor.Compute(b1_x2[i], b2_x2[i], b1_x1[i], b2_x1[i]);
    const T b_y_min_max = functor.Compute(b1_y2[i], b2_y2[i], b1_y1[i], b2_y1[i]);
    const T b_x_min_max_inter = b_x_min_max * inter_diff_i;
    const T b_y_min_max_inter = b_y_min_max * inter_diff_i;

    b1_x1_diff[i] = static_cast<T>(0.0);
    b1_x2_diff[i] = static_cast<T>(0.0);
    b2_x1_diff[i] = static_cast<T>(0.0);
    b2_x2_diff[i] = static_cast<T>(0.0);
    b1_y1_diff[i] = static_cast<T>(0.0);
    b1_y2_diff[i] = static_cast<T>(0.0);
    b2_y1_diff[i] = static_cast<T>(0.0);
    b2_y2_diff[i] = static_cast<T>(0.0);

    if (b_x_min_max > static_cast<T>(0.0) && b_y_min_max > static_cast<T>(0.0)) {
      if (b1_x1[i] >= b2_x1[i]) { b1_x1_diff[i] = static_cast<T>(-1.0) * b_y_min_max_inter; }
      if (b1_x1[i] <= b2_x1[i]) { b2_x1_diff[i] = static_cast<T>(-1.0) * b_y_min_max_inter; }
      if (b1_x2[i] <= b2_x2[i]) { b1_x2_diff[i] = b_y_min_max_inter; }
      if (b1_x2[i] >= b2_x2[i]) { b2_x2_diff[i] = b_y_min_max_inter; }

      if (b1_y1[i] >= b2_y1[i]) { b1_y1_diff[i] = static_cast<T>(-1.0) * b_x_min_max_inter; }
      if (b1_y1[i] <= b2_y1[i]) { b2_y1_diff[i] = static_cast<T>(-1.0) * b_x_min_max_inter; }
      if (b1_y2[i] <= b2_y2[i]) { b1_y2_diff[i] = b_x_min_max_inter; }
      if (b1_y2[i] >= b2_y2[i]) { b2_y2_diff[i] = b_x_min_max_inter; }
    }
  }
}

template<typename FUNCTOR, typename T>
__global__ void FusedGetIntersectionAreaForward(FUNCTOR functor, const int n, const T* b1_x1,
                                                const T* b1_x2, const T* b2_x1, const T* b2_x2,
                                                const T* b1_y1, const T* b1_y2, const T* b2_y1,
                                                const T* b2_y2, T* inter) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T b_x_min_max = functor.Compute(b1_x2[i], b2_x2[i], b1_x1[i], b2_x1[i]);
    const T b_y_min_max = functor.Compute(b1_y2[i], b2_y2[i], b1_y1[i], b2_y1[i]);
    inter[i] = static_cast<T>(0.0);
    if (b_x_min_max > static_cast<T>(0.0) && b_y_min_max > static_cast<T>(0.0)) {
      inter[i] = b_x_min_max * b_y_min_max;
    }
  }
}

}  // namespace

template<typename T>
class FusedGetIntersectionAreaKernel final : public user_op::OpKernel {
 public:
  FusedGetIntersectionAreaKernel() = default;
  ~FusedGetIntersectionAreaKernel() = default;

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

    user_op::Tensor* inter = ctx->Tensor4ArgNameAndIndex("inter", 0);

    const int64_t elem_cnt = b1_x2->shape_view().elem_cnt();

    MinMaxDeltaFunctor<T> min_max_delta_functor{};

    RUN_CUDA_KERNEL((FusedGetIntersectionAreaForward<decltype(min_max_delta_functor), T>),
                    ctx->stream(), elem_cnt, min_max_delta_functor, elem_cnt, b1_x1->dptr<T>(),
                    b1_x2->dptr<T>(), b2_x1->dptr<T>(), b2_x2->dptr<T>(), b1_y1->dptr<T>(),
                    b1_y2->dptr<T>(), b2_y1->dptr<T>(), b2_y2->dptr<T>(), inter->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_INTERSECTION_AREA_CUDA_KERNEL(dtype)        \
  REGISTER_USER_KERNEL("fused_get_intersection_area")                  \
      .SetCreateFn<FusedGetIntersectionAreaKernel<dtype>>()            \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("inter", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_INTERSECTION_AREA_CUDA_KERNEL(float)
REGISTER_FUSED_GET_INTERSECTION_AREA_CUDA_KERNEL(double)
REGISTER_FUSED_GET_INTERSECTION_AREA_CUDA_KERNEL(half)

template<typename T>
class FusedGetIntersectionAreaGradKernel final : public user_op::OpKernel {
 public:
  FusedGetIntersectionAreaGradKernel() = default;
  ~FusedGetIntersectionAreaGradKernel() = default;

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

    user_op::Tensor* inter_diff = ctx->Tensor4ArgNameAndIndex("inter_diff", 0);

    user_op::Tensor* b1_x1_diff = ctx->Tensor4ArgNameAndIndex("b1_x1_diff", 0);
    user_op::Tensor* b1_x2_diff = ctx->Tensor4ArgNameAndIndex("b1_x2_diff", 0);
    user_op::Tensor* b2_x1_diff = ctx->Tensor4ArgNameAndIndex("b2_x1_diff", 0);
    user_op::Tensor* b2_x2_diff = ctx->Tensor4ArgNameAndIndex("b2_x2_diff", 0);
    user_op::Tensor* b1_y1_diff = ctx->Tensor4ArgNameAndIndex("b1_y1_diff", 0);
    user_op::Tensor* b1_y2_diff = ctx->Tensor4ArgNameAndIndex("b1_y2_diff", 0);
    user_op::Tensor* b2_y1_diff = ctx->Tensor4ArgNameAndIndex("b2_y1_diff", 0);
    user_op::Tensor* b2_y2_diff = ctx->Tensor4ArgNameAndIndex("b2_y2_diff", 0);

    const int64_t elem_cnt = b1_x1->shape_view().elem_cnt();

    MinMaxDeltaFunctor<T> min_max_delta_functor{};

    RUN_CUDA_KERNEL((FusedGetIntersectionAreaBackward<decltype(min_max_delta_functor), T>),
                    ctx->stream(), elem_cnt, min_max_delta_functor, elem_cnt, b1_x1->dptr<T>(),
                    b1_x2->dptr<T>(), b2_x1->dptr<T>(), b2_x2->dptr<T>(), b1_y1->dptr<T>(),
                    b1_y2->dptr<T>(), b2_y1->dptr<T>(), b2_y2->dptr<T>(), inter_diff->dptr<T>(),
                    b1_x1_diff->mut_dptr<T>(), b1_x2_diff->mut_dptr<T>(), b2_x1_diff->mut_dptr<T>(),
                    b2_x2_diff->mut_dptr<T>(), b1_y1_diff->mut_dptr<T>(), b1_y2_diff->mut_dptr<T>(),
                    b2_y1_diff->mut_dptr<T>(), b2_y2_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_INTERSECTION_AREA_GRAD_CUDA_KERNEL(dtype)   \
  REGISTER_USER_KERNEL("fused_get_intersection_area_grad")             \
      .SetCreateFn<FusedGetIntersectionAreaGradKernel<dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("b1_x1_diff", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_INTERSECTION_AREA_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_GET_INTERSECTION_AREA_GRAD_CUDA_KERNEL(double)
REGISTER_FUSED_GET_INTERSECTION_AREA_GRAD_CUDA_KERNEL(half)

}  // namespace oneflow
