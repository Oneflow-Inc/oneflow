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
__global__ void FusedGetIouForward(const int n, const T* w1, const T* h1, const T* w2, const T* h2,
                                   const T* inter, T* iou, const float eps) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T inter_i = inter[i];
    iou[i] = inter_i / (w1[i] * h1[i] + w2[i] * h2[i] - inter_i + static_cast<T>(eps));
  }
}

template<typename T>
__global__ void FusedGetIouBackward(const int n, const T* diou, const T* w1, const T* h1,
                                    const T* w2, const T* h2, const T* inter, T* dw1, T* dh1,
                                    T* dinter, const float eps) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T w1_i = w1[i], h1_i = h1[i], w2_i = w2[i], h2_i = h2[i], inter_i = inter[i],
            diou_i = diou[i];
    const T w_h_eps = w1_i * h1_i + w2_i * h2_i + static_cast<T>(eps);
    const T w_h_eps_inter_diff = w_h_eps - inter_i;
    const T w_h_eps_inter_diff_square = w_h_eps_inter_diff * w_h_eps_inter_diff;
    const T common_for_dwh = -inter_i * diou_i / w_h_eps_inter_diff_square;
    dinter[i] = w_h_eps * diou_i / w_h_eps_inter_diff_square;
    dw1[i] = h1_i * common_for_dwh;
    dh1[i] = w1_i * common_for_dwh;
  }
}
};  // namespace

template<typename T>
class FusedGetIouGpuKernel final : public user_op::OpKernel {
 public:
  FusedGetIouGpuKernel() = default;
  ~FusedGetIouGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* w1 = ctx->Tensor4ArgNameAndIndex("w1", 0);
    const user_op::Tensor* h1 = ctx->Tensor4ArgNameAndIndex("h1", 0);
    const user_op::Tensor* w2 = ctx->Tensor4ArgNameAndIndex("w2", 0);
    const user_op::Tensor* h2 = ctx->Tensor4ArgNameAndIndex("h2", 0);
    const user_op::Tensor* inter = ctx->Tensor4ArgNameAndIndex("inter", 0);

    user_op::Tensor* iou = ctx->Tensor4ArgNameAndIndex("iou", 0);

    float eps = ctx->Attr<float>("eps");

    const int32_t elem_cnt = w1->shape_view().elem_cnt();
    RUN_CUDA_KERNEL((FusedGetIouForward<T>), ctx->stream(), elem_cnt, elem_cnt, w1->dptr<T>(),
                    h1->dptr<T>(), w2->dptr<T>(), h2->dptr<T>(), inter->dptr<T>(),
                    iou->mut_dptr<T>(), eps);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_IOU_CUDA_KERNEL(dtype)                      \
  REGISTER_USER_KERNEL("fused_get_iou")                                \
      .SetCreateFn<FusedGetIouGpuKernel<dtype>>()                      \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("iou", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_IOU_CUDA_KERNEL(float)
REGISTER_FUSED_GET_IOU_CUDA_KERNEL(half)
REGISTER_FUSED_GET_IOU_CUDA_KERNEL(double)

template<typename T>
class FusedGetIouGradGpuKernel final : public user_op::OpKernel {
 public:
  FusedGetIouGradGpuKernel() = default;
  ~FusedGetIouGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* diou = ctx->Tensor4ArgNameAndIndex("diou", 0);
    const user_op::Tensor* w1 = ctx->Tensor4ArgNameAndIndex("w1", 0);
    const user_op::Tensor* h1 = ctx->Tensor4ArgNameAndIndex("h1", 0);
    const user_op::Tensor* w2 = ctx->Tensor4ArgNameAndIndex("w2", 0);
    const user_op::Tensor* h2 = ctx->Tensor4ArgNameAndIndex("h2", 0);
    const user_op::Tensor* inter = ctx->Tensor4ArgNameAndIndex("inter", 0);

    user_op::Tensor* dw1 = ctx->Tensor4ArgNameAndIndex("dw1", 0);
    user_op::Tensor* dh1 = ctx->Tensor4ArgNameAndIndex("dh1", 0);
    user_op::Tensor* dinter = ctx->Tensor4ArgNameAndIndex("dinter", 0);

    float eps = ctx->Attr<float>("eps");

    const int32_t elem_cnt = diou->shape_view().elem_cnt();

    RUN_CUDA_KERNEL((FusedGetIouBackward<T>), ctx->stream(), elem_cnt, elem_cnt, diou->dptr<T>(),
                    w1->dptr<T>(), h1->dptr<T>(), w2->dptr<T>(), h2->dptr<T>(), inter->dptr<T>(),
                    dw1->mut_dptr<T>(), dh1->mut_dptr<T>(), dinter->mut_dptr<T>(), eps);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_IOU_GRAD_CUDA_KERNEL(dtype)                 \
  REGISTER_USER_KERNEL("fused_get_iou_grad")                           \
      .SetCreateFn<FusedGetIouGradGpuKernel<dtype>>()                  \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("diou", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_IOU_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_GET_IOU_GRAD_CUDA_KERNEL(half)
REGISTER_FUSED_GET_IOU_GRAD_CUDA_KERNEL(double)

}  // namespace oneflow
