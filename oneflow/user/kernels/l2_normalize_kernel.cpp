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
static void L2NormalizeForward(const int32_t n, const int32_t c, const int32_t d, const T epsilon,
                               const T* in, T* square_x_sum, T* out) {
  for (int32_t i = 0; i < n; i++) {
    const int32_t offset = (i / d) * d * c + (i % d);
    for (int32_t j = 0; j < c; j++) {
      const T x = in[offset + j * d];
      square_x_sum[i] += x * x;
    }
    const T norm = std::sqrt(std::max(square_x_sum[i], epsilon));
    for (int32_t j = 0; j < c; j++) {
      const int32_t index = offset + j * d;
      out[index] = in[index] / norm;
    }
  }
}

template<typename T>
static void L2NormalizeBackward(const int32_t n, const int32_t c, const int32_t d, const T epsilon,
                                const T* out, const T* out_diff, const T* square_x_sum,
                                T* in_diff) {
  for (int32_t i = 0; i < n; i++) {
    const T norm = std::sqrt(std::max(square_x_sum[i], epsilon));
    const int32_t offset = (i / d) * d * c + (i % d);
    if (square_x_sum[i] >= epsilon) {
      T y_dy_inner_prod = GetZeroVal<T>();
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = offset + j * d;
        y_dy_inner_prod += out_diff[index] * out[index];
      }
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = offset + j * d;
        in_diff[index] = (1 / norm) * (out_diff[index] - y_dy_inner_prod * out[index]);
      }
    } else {
      for (int32_t j = 0; j < c; j++) {
        const int32_t index = offset + j * d;
        in_diff[index] = (1 / norm) * out_diff[index];
      }
    }
  }
}

}  // namespace

template<typename T>
class CpuL2NormalizeKernel final : public user_op::OpKernel {
 public:
  CpuL2NormalizeKernel() = default;
  ~CpuL2NormalizeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* square_x_sum = ctx->Tensor4ArgNameAndIndex("square_x_sum", 0);
    const float epsilon = ctx->Attr<float>("epsilon");
    int32_t axis = ctx->Attr<int32_t>("axis");
    int32_t c = x->shape().At(axis);
    int32_t n = x->shape().elem_cnt() / c;
    int32_t d = x->shape().Count(axis + 1);

    size_t square_x_sum_byte_size = square_x_sum->shape().elem_cnt() * sizeof(T);
    Memset<DeviceType::kCPU>(ctx->device_ctx(), square_x_sum->mut_dptr(), 0,
                             square_x_sum_byte_size);
    L2NormalizeForward<T>(n, c, d, static_cast<T>(epsilon), x->dptr<T>(),
                          square_x_sum->mut_dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_L2_NORMALIZE_KERNEL(dtype)           \
  REGISTER_USER_KERNEL("l2_normalize")                    \
      .SetCreateFn<CpuL2NormalizeKernel<dtype>>()         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_CPU_L2_NORMALIZE_KERNEL(float)

template<typename T>
class CpuL2NormalizeGradKernel final : public user_op::OpKernel {
 public:
  CpuL2NormalizeGradKernel() = default;
  ~CpuL2NormalizeGradKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const user_op::Tensor* dy = ctx->Tensor4ArgNameAndIndex("dy", 0);
    const user_op::Tensor* square_x_sum = ctx->Tensor4ArgNameAndIndex("square_x_sum", 0);
    user_op::Tensor* dx = ctx->Tensor4ArgNameAndIndex("dx", 0);
    const float epsilon = ctx->Attr<float>("epsilon");
    int32_t axis = ctx->Attr<int32_t>("axis");
    int32_t c = dy->shape().At(axis);
    int32_t n = dy->shape().elem_cnt() / c;
    int32_t d = dy->shape().Count(axis + 1);
    L2NormalizeBackward<T>(n, c, d, static_cast<T>(epsilon), y->dptr<T>(), dy->dptr<T>(),
                           square_x_sum->dptr<T>(), dx->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_L2_NORMALIZE_GRAD_KERNEL(dtype)      \
  REGISTER_USER_KERNEL("l2_normalize_grad")               \
      .SetCreateFn<CpuL2NormalizeGradKernel<dtype>>()     \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu") \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_CPU_L2_NORMALIZE_GRAD_KERNEL(float)

}  // namespace oneflow
