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
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {
template<typename T>
__global__ void FusedGetBounddingBoxesCoordForward(const int n, const T* x1, const T* y1,
                                                   const T* w1, const T* h1, const T* x2,
                                                   const T* y2, const T* w2, const T* h2, T* b1_x1,
                                                   T* b1_x2, T* b1_y1, T* b1_y2, T* b2_x1, T* b2_x2,
                                                   T* b2_y1, T* b2_y2) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T w1_ = w1[i] / static_cast<T>(2.0);
    const T h1_ = h1[i] / static_cast<T>(2.0);
    const T w2_ = w2[i] / static_cast<T>(2.0);
    const T h2_ = h2[i] / static_cast<T>(2.0);
    const T x1_i = x1[i], y1_i = y1[i], x2_i = x2[i], y2_i = y2[i];
    b1_x1[i] = x1_i - w1_;
    b1_x2[i] = x1_i + w1_;
    b1_y1[i] = y1_i - h1_;
    b1_y2[i] = y1_i + h1_;
    b2_x1[i] = x2_i - w2_;
    b2_x2[i] = x2_i + w2_;
    b2_y1[i] = y2_i - h2_;
    b2_y2[i] = y2_i + h2_;
  }
}

template<typename T>
__global__ void FusedGetBounddingBoxesCoordBackward(
    const int n, const T* b1_x1_diff, const T* b1_x2_diff, const T* b1_y1_diff, const T* b1_y2_diff,
    const T* b2_x1_diff, const T* b2_x2_diff, const T* b2_y1_diff, const T* b2_y2_diff, T* x1_diff,
    T* y1_diff, T* w1_diff, T* h1_diff, T* x2_diff, T* y2_diff, T* w2_diff, T* h2_diff) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T b1_x1_diff_i = b1_x1_diff[i];
    const T b1_x2_diff_i = b1_x2_diff[i];
    const T b1_y1_diff_i = b1_y1_diff[i];
    const T b1_y2_diff_i = b1_y2_diff[i];
    const T b2_x1_diff_i = b2_x1_diff[i];
    const T b2_x2_diff_i = b2_x2_diff[i];
    const T b2_y2_diff_i = b2_y2_diff[i];
    const T b2_y1_diff_i = b2_y1_diff[i];
    x1_diff[i] = b1_x1_diff_i + b1_x2_diff_i;
    y1_diff[i] = b1_y1_diff_i + b1_y2_diff_i;
    w1_diff[i] = (b1_x2_diff_i - b1_x1_diff_i) / static_cast<T>(2.0);
    h1_diff[i] = (b1_y2_diff_i - b1_y1_diff_i) / static_cast<T>(2.0);
    x2_diff[i] = b2_x1_diff_i + b2_x2_diff_i;
    y2_diff[i] = b2_y1_diff_i + b2_y2_diff_i;
    w2_diff[i] = (b2_x2_diff_i - b2_x1_diff_i) / static_cast<T>(2.0);
    h2_diff[i] = (b2_y2_diff_i - b2_y1_diff_i) / static_cast<T>(2.0);
  }
}
};  // namespace

template<typename T>
class FusedGetBounddingBoxesCoordGpuKernel final : public user_op::OpKernel {
 public:
  FusedGetBounddingBoxesCoordGpuKernel() = default;
  ~FusedGetBounddingBoxesCoordGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x1 = ctx->Tensor4ArgNameAndIndex("x1", 0);
    const user_op::Tensor* y1 = ctx->Tensor4ArgNameAndIndex("y1", 0);
    const user_op::Tensor* w1 = ctx->Tensor4ArgNameAndIndex("w1", 0);
    const user_op::Tensor* h1 = ctx->Tensor4ArgNameAndIndex("h1", 0);
    const user_op::Tensor* x2 = ctx->Tensor4ArgNameAndIndex("x2", 0);
    const user_op::Tensor* y2 = ctx->Tensor4ArgNameAndIndex("y2", 0);
    const user_op::Tensor* w2 = ctx->Tensor4ArgNameAndIndex("w2", 0);
    const user_op::Tensor* h2 = ctx->Tensor4ArgNameAndIndex("h2", 0);

    user_op::Tensor* b1_x1 = ctx->Tensor4ArgNameAndIndex("b1_x1", 0);
    user_op::Tensor* b1_x2 = ctx->Tensor4ArgNameAndIndex("b1_x2", 0);
    user_op::Tensor* b1_y1 = ctx->Tensor4ArgNameAndIndex("b1_y1", 0);
    user_op::Tensor* b1_y2 = ctx->Tensor4ArgNameAndIndex("b1_y2", 0);
    user_op::Tensor* b2_x1 = ctx->Tensor4ArgNameAndIndex("b2_x1", 0);
    user_op::Tensor* b2_x2 = ctx->Tensor4ArgNameAndIndex("b2_x2", 0);
    user_op::Tensor* b2_y1 = ctx->Tensor4ArgNameAndIndex("b2_y1", 0);
    user_op::Tensor* b2_y2 = ctx->Tensor4ArgNameAndIndex("b2_y2", 0);

    const int32_t elem_cnt = x1->shape_view().elem_cnt();
    RUN_CUDA_KERNEL((FusedGetBounddingBoxesCoordForward<T>), ctx->stream(), elem_cnt, elem_cnt,
                    x1->dptr<T>(), y1->dptr<T>(), w1->dptr<T>(), h1->dptr<T>(), x2->dptr<T>(),
                    y2->dptr<T>(), w2->dptr<T>(), h2->dptr<T>(), b1_x1->mut_dptr<T>(),
                    b1_x2->mut_dptr<T>(), b1_y1->mut_dptr<T>(), b1_y2->mut_dptr<T>(),
                    b2_x1->mut_dptr<T>(), b2_x2->mut_dptr<T>(), b2_y1->mut_dptr<T>(),
                    b2_y2->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_BOUNDDING_BOXES_COORD_CUDA_KERNEL(dtype)    \
  REGISTER_USER_KERNEL("fused_get_boundding_boxes_coord")              \
      .SetCreateFn<FusedGetBounddingBoxesCoordGpuKernel<dtype>>()      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("b1_x1", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_BOUNDDING_BOXES_COORD_CUDA_KERNEL(float)
REGISTER_FUSED_GET_BOUNDDING_BOXES_COORD_CUDA_KERNEL(half)
REGISTER_FUSED_GET_BOUNDDING_BOXES_COORD_CUDA_KERNEL(double)

template<typename T>
class FusedGetBounddingBoxesCoordGradGpuKernel final : public user_op::OpKernel {
 public:
  FusedGetBounddingBoxesCoordGradGpuKernel() = default;
  ~FusedGetBounddingBoxesCoordGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* b1_x1_diff = ctx->Tensor4ArgNameAndIndex("b1_x1_diff", 0);
    const user_op::Tensor* b1_x2_diff = ctx->Tensor4ArgNameAndIndex("b1_x2_diff", 0);
    const user_op::Tensor* b1_y1_diff = ctx->Tensor4ArgNameAndIndex("b1_y1_diff", 0);
    const user_op::Tensor* b1_y2_diff = ctx->Tensor4ArgNameAndIndex("b1_y2_diff", 0);
    const user_op::Tensor* b2_x1_diff = ctx->Tensor4ArgNameAndIndex("b2_x1_diff", 0);
    const user_op::Tensor* b2_x2_diff = ctx->Tensor4ArgNameAndIndex("b2_x2_diff", 0);
    const user_op::Tensor* b2_y1_diff = ctx->Tensor4ArgNameAndIndex("b2_y1_diff", 0);
    const user_op::Tensor* b2_y2_diff = ctx->Tensor4ArgNameAndIndex("b2_y2_diff", 0);

    user_op::Tensor* x1_diff = ctx->Tensor4ArgNameAndIndex("x1_diff", 0);
    user_op::Tensor* y1_diff = ctx->Tensor4ArgNameAndIndex("y1_diff", 0);
    user_op::Tensor* w1_diff = ctx->Tensor4ArgNameAndIndex("w1_diff", 0);
    user_op::Tensor* h1_diff = ctx->Tensor4ArgNameAndIndex("h1_diff", 0);
    user_op::Tensor* x2_diff = ctx->Tensor4ArgNameAndIndex("x2_diff", 0);
    user_op::Tensor* y2_diff = ctx->Tensor4ArgNameAndIndex("y2_diff", 0);
    user_op::Tensor* w2_diff = ctx->Tensor4ArgNameAndIndex("w2_diff", 0);
    user_op::Tensor* h2_diff = ctx->Tensor4ArgNameAndIndex("h2_diff", 0);

    const int32_t elem_cnt = b1_x1_diff->shape_view().elem_cnt();
    RUN_CUDA_KERNEL((FusedGetBounddingBoxesCoordBackward<T>), ctx->stream(), elem_cnt, elem_cnt,
                    b1_x1_diff->dptr<T>(), b1_x2_diff->dptr<T>(), b1_y1_diff->dptr<T>(),
                    b1_y2_diff->dptr<T>(), b2_x1_diff->dptr<T>(), b2_x2_diff->dptr<T>(),
                    b2_y1_diff->dptr<T>(), b2_y2_diff->dptr<T>(), x1_diff->mut_dptr<T>(),
                    y1_diff->mut_dptr<T>(), w1_diff->mut_dptr<T>(), h1_diff->mut_dptr<T>(),
                    x2_diff->mut_dptr<T>(), y2_diff->mut_dptr<T>(), w2_diff->mut_dptr<T>(),
                    h2_diff->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_BOUNDDING_BOXES_COORD_GRAD_CUDA_KERNEL(dtype) \
  REGISTER_USER_KERNEL("fused_get_boundding_boxes_coord_grad")           \
      .SetCreateFn<FusedGetBounddingBoxesCoordGradGpuKernel<dtype>>()    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA)   \
                       && (user_op::HobDataType("b1_x1_diff", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_BOUNDDING_BOXES_COORD_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_GET_BOUNDDING_BOXES_COORD_GRAD_CUDA_KERNEL(half)
REGISTER_FUSED_GET_BOUNDDING_BOXES_COORD_GRAD_CUDA_KERNEL(double)

}  // namespace oneflow
