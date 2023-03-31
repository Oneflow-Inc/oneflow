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
#include "oneflow/core/common/math_util.h"

namespace oneflow {

namespace {

template<typename T>
__global__ void FusedGetConvexDiagonalSquaredForward(const int n, const T* b1_x1, const T* b1_x2,
                                                     const T* b2_x1, const T* b2_x2, const T* b1_y1,
                                                     const T* b1_y2, const T* b2_y1, const T* b2_y2,
                                                     T* c2, const float eps) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T cw = DeviceMax(b1_x2[i], b2_x2[i]) - DeviceMin(b1_x1[i], b2_x1[i]);
    const T ch = DeviceMax(b1_y2[i], b2_y2[i]) - DeviceMin(b1_y1[i], b2_y1[i]);
    c2[i] = cw * cw + ch * ch + static_cast<T>(eps);
  }
}

template<typename T>
__global__ void FusedGetConvexDiagonalSquaredBackward(
    const int n, const T* b1_x1, const T* b1_x2, const T* b2_x1, const T* b2_x2, const T* b1_y1,
    const T* b1_y2, const T* b2_y1, const T* b2_y2, const T* c2_diff, T* b1_x1_diff, T* b1_x2_diff,
    T* b2_x1_diff, T* b2_x2_diff, T* b1_y1_diff, T* b1_y2_diff, T* b2_y1_diff, T* b2_y2_diff,
    const float eps) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T zero = static_cast<T>(0), one = static_cast<T>(1);
    const T cw = DeviceMax(b1_x2[i], b2_x2[i]) - DeviceMin(b1_x1[i], b2_x1[i]);
    const T ch = DeviceMax(b1_y2[i], b2_y2[i]) - DeviceMin(b1_y1[i], b2_y1[i]);
    const T c2_diff_cw = static_cast<T>(2) * cw * c2_diff[i];
    const T c2_diff_ch = static_cast<T>(2) * ch * c2_diff[i];
    b1_x2_diff[i] = c2_diff_cw * (b1_x2[i] > b2_x2[i] ? one : zero);
    b2_x2_diff[i] = c2_diff_cw * (b1_x2[i] > b2_x2[i] ? zero : one);
    b1_x1_diff[i] = -c2_diff_cw * (b1_x1[i] < b2_x1[i] ? one : zero);
    b2_x1_diff[i] = -c2_diff_cw * (b1_x1[i] < b2_x1[i] ? zero : one);
    b1_y2_diff[i] = c2_diff_ch * (b1_y2[i] > b2_y2[i] ? one : zero);
    b2_y2_diff[i] = c2_diff_ch * (b1_y2[i] > b2_y2[i] ? zero : one);
    b1_y1_diff[i] = -c2_diff_ch * (b1_y1[i] < b2_y1[i] ? one : zero);
    b2_y1_diff[i] = -c2_diff_ch * (b1_y1[i] < b2_y1[i] ? zero : one);
  }
}

}  // namespace

template<typename T>
class FusedGetConvexDiagonalSquaredKernel final : public user_op::OpKernel {
 public:
  FusedGetConvexDiagonalSquaredKernel() = default;
  ~FusedGetConvexDiagonalSquaredKernel() = default;

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

    user_op::Tensor* c2 = ctx->Tensor4ArgNameAndIndex("c2", 0);
    const float eps = ctx->Attr<float>("eps");

    const int64_t elem_cnt = b1_x1->shape_view().elem_cnt();

    RUN_CUDA_KERNEL((FusedGetConvexDiagonalSquaredForward<T>), ctx->stream(), elem_cnt, elem_cnt,
                    b1_x1->dptr<T>(), b1_x2->dptr<T>(), b2_x1->dptr<T>(), b2_x2->dptr<T>(),
                    b1_y1->dptr<T>(), b1_y2->dptr<T>(), b2_y1->dptr<T>(), b2_y2->dptr<T>(),
                    c2->mut_dptr<T>(), eps);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_CONVEX_DIAGOAL_SQUARED_CUDA_KERNEL(dtype)   \
  REGISTER_USER_KERNEL("fused_get_convex_diagonal_squared")            \
      .SetCreateFn<FusedGetConvexDiagonalSquaredKernel<dtype>>()       \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("b1_x1", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_CONVEX_DIAGOAL_SQUARED_CUDA_KERNEL(float)
REGISTER_FUSED_GET_CONVEX_DIAGOAL_SQUARED_CUDA_KERNEL(double)
REGISTER_FUSED_GET_CONVEX_DIAGOAL_SQUARED_CUDA_KERNEL(half)

template<typename T>
class FusedGetConvexDiagonalSquaredGradKernel final : public user_op::OpKernel {
 public:
  FusedGetConvexDiagonalSquaredGradKernel() = default;
  ~FusedGetConvexDiagonalSquaredGradKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* c2_diff = ctx->Tensor4ArgNameAndIndex("c2_diff", 0);
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

    const float eps = ctx->Attr<float>("eps");
    const int64_t elem_cnt = b1_x1_diff->shape_view().elem_cnt();

    RUN_CUDA_KERNEL((FusedGetConvexDiagonalSquaredBackward<T>), ctx->stream(), elem_cnt, elem_cnt,
                    b1_x1->dptr<T>(), b1_x2->dptr<T>(), b2_x1->dptr<T>(), b2_x2->dptr<T>(),
                    b1_y1->dptr<T>(), b1_y2->dptr<T>(), b2_y1->dptr<T>(), b2_y2->dptr<T>(),
                    c2_diff->dptr<T>(), b1_x1_diff->mut_dptr<T>(), b1_x2_diff->mut_dptr<T>(),
                    b2_x1_diff->mut_dptr<T>(), b2_x2_diff->mut_dptr<T>(), b1_y1_diff->mut_dptr<T>(),
                    b1_y2_diff->mut_dptr<T>(), b2_y1_diff->mut_dptr<T>(), b2_y2_diff->mut_dptr<T>(),
                    eps);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_CONVEX_DIAGOAL_SQUARED_GRAD_CUDA_KERNEL(dtype) \
  REGISTER_USER_KERNEL("fused_get_convex_diagonal_squared_grad")          \
      .SetCreateFn<FusedGetConvexDiagonalSquaredGradKernel<dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)    \
                       && (user_op::HobDataType("b1_x1", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_CONVEX_DIAGOAL_SQUARED_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_GET_CONVEX_DIAGOAL_SQUARED_GRAD_CUDA_KERNEL(double)
REGISTER_FUSED_GET_CONVEX_DIAGOAL_SQUARED_GRAD_CUDA_KERNEL(half)

}  // namespace oneflow
