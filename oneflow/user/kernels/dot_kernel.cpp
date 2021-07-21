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

template<DeviceType device_type, typename T>
struct DotKernelCalculation {
  static void dot(DeviceCtx* ctx, const int64_t n, const T* x,  const T* y,
                   T* out);
};

template<typename T>
struct DotKernelCalculation<DeviceType::kCPU, T> {
  static void dot(DeviceCtx* ctx, const int64_t n, const T* x,  const T* y,
                   T* out) {
    *out = cblas_dot<T>(n, x, 1, y, 1);
  }
};
template<typename T>
struct DotKernelCalculation<DeviceType::kGPU, T> {
  static void dot(DeviceCtx* ctx, const int64_t n, const T* x,  const T* y,
                  T* out) {
    NewKernelUtil<DeviceType::kGPU>::OFDot(ctx, n, x, 1, y, 1, out);
  }
};

template<DeviceType device_type, typename T>
class DotKernel final : public user_op::OpKernel {
 public:
  DotKernel() = default;
  ~DotKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = x->shape().elem_cnt();
    CHECK(n <= INT_MAX);
    DotKernelCalculation<device_type, T>::dot(ctx->device_ctx(), n, x->dptr<T>(), y->dptr<T>(),
                                        out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DOT_KERNEL(device, dtype)                                             \
  REGISTER_USER_KERNEL("dot").SetCreateFn<DotKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                              \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_DOT_KERNEL(DeviceType::kCPU, float)
REGISTER_DOT_KERNEL(DeviceType::kCPU, double)

#ifdef WITH_CUDA
REGISTER_DOT_KERNEL(DeviceType::kGPU, float)
REGISTER_DOT_KERNEL(DeviceType::kGPU, double)
#endif

}  // namespace

}  // namespace oneflow
