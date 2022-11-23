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
__global__ void FusedGetCiouResultForward(const int n, const T* v, const T* iou, const T* rho2,
                                          const T* c2, T* y, T* alpha, float eps) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T v_i = v[i];
    const T iou_i = iou[i];
    const T alpha_i = v_i / (v_i - iou_i + static_cast<T>(1.0 + eps));
    y[i] = iou_i - (rho2[i] / c2[i] + v_i * alpha_i);
    alpha[i] = alpha_i;
  }
}

template<typename T>
__global__ void FusedGetCiouResultBackward(const int n, const T* dy, const T* alpha, const T* rho2,
                                           const T* c2, T* dv, T* diou, T* drho2, T* dc2) {
  CUDA_1D_KERNEL_LOOP(i, n) {
    const T c2_i = c2[i];
    const T dy_i = dy[i];
    dv[i] = -alpha[i] * dy_i;
    diou[i] = dy_i;
    drho2[i] = -dy_i / c2[i];
    dc2[i] = rho2[i] / (c2_i * c2_i) * dy_i;
  }
}
};  // namespace

template<typename T>
class FusedGetCiouResultGpuKernel final : public user_op::OpKernel {
 public:
  FusedGetCiouResultGpuKernel() = default;
  ~FusedGetCiouResultGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* v = ctx->Tensor4ArgNameAndIndex("v", 0);
    const user_op::Tensor* iou = ctx->Tensor4ArgNameAndIndex("iou", 0);
    const user_op::Tensor* rho2 = ctx->Tensor4ArgNameAndIndex("rho2", 0);
    const user_op::Tensor* c2 = ctx->Tensor4ArgNameAndIndex("c2", 0);

    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);

    float eps = ctx->Attr<float>("eps");

    const int32_t elem_cnt = v->shape_view().elem_cnt();
    RUN_CUDA_KERNEL((FusedGetCiouResultForward<T>), ctx->stream(), elem_cnt, elem_cnt, v->dptr<T>(),
                    iou->dptr<T>(), rho2->dptr<T>(), c2->dptr<T>(), y->mut_dptr<T>(),
                    alpha->mut_dptr<T>(), eps);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_CIOU_RESULT_CUDA_KERNEL(dtype)              \
  REGISTER_USER_KERNEL("fused_get_ciou_result")                        \
      .SetCreateFn<FusedGetCiouResultGpuKernel<dtype>>()               \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("v", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_CIOU_RESULT_CUDA_KERNEL(float)
REGISTER_FUSED_GET_CIOU_RESULT_CUDA_KERNEL(half)
REGISTER_FUSED_GET_CIOU_RESULT_CUDA_KERNEL(double)

template<typename T>
class FusedGetCiouResultGradGpuKernel final : public user_op::OpKernel {
 public:
  FusedGetCiouResultGradGpuKernel() = default;
  ~FusedGetCiouResultGradGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* alpha = ctx->Tensor4ArgNameAndIndex("alpha", 0);
    const user_op::Tensor* rho2 = ctx->Tensor4ArgNameAndIndex("rho2", 0);
    const user_op::Tensor* c2 = ctx->Tensor4ArgNameAndIndex("c2", 0);

    user_op::Tensor* dv = ctx->Tensor4ArgNameAndIndex("dv", 0);
    user_op::Tensor* diou = ctx->Tensor4ArgNameAndIndex("diou", 0);
    user_op::Tensor* drho2 = ctx->Tensor4ArgNameAndIndex("drho2", 0);
    user_op::Tensor* dc2 = ctx->Tensor4ArgNameAndIndex("dc2", 0);

    const int32_t elem_cnt = dy->shape_view().elem_cnt();
    RUN_CUDA_KERNEL((FusedGetCiouResultBackward<T>), ctx->stream(), elem_cnt, elem_cnt,
                    dy->dptr<T>(), alpha->dptr<T>(), rho2->dptr<T>(), c2->dptr<T>(),
                    dv->mut_dptr<T>(), diou->mut_dptr<T>(), drho2->mut_dptr<T>(),
                    dc2->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FUSED_GET_CIOU_RESULT_GRAD_CUDA_KERNEL(dtype)         \
  REGISTER_USER_KERNEL("fused_get_ciou_result_grad")                   \
      .SetCreateFn<FusedGetCiouResultGradGpuKernel<dtype>>()           \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCUDA) \
                       && (user_op::HobDataType("dy", 0) == GetDataType<dtype>::value));

REGISTER_FUSED_GET_CIOU_RESULT_GRAD_CUDA_KERNEL(float)
REGISTER_FUSED_GET_CIOU_RESULT_GRAD_CUDA_KERNEL(half)
REGISTER_FUSED_GET_CIOU_RESULT_GRAD_CUDA_KERNEL(double)

}  // namespace oneflow
