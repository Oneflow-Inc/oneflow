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
T dot_naive(const int64_t n, const T* x, int64_t incx, const T* y, int64_t incy) {
  T sum = (T)(0);
  
  for (int64_t i = 0; i < n; i++) { sum += x[i * incx] * y[i * incy]; }
  return sum;
}

template<typename T>
class DotCpuKernel final : public user_op::OpKernel {
 public:
  DotCpuKernel() = default;
  ~DotCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    T* out_ptr = out_tensor->mut_dptr<T>();
    *out_ptr = dot_naive<T>(x->shape().elem_cnt(), x->dptr<T>(), 1, y->dptr<T>(), 1);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_DOT_CPU_KERNEL(device, dtype)                                     \
  REGISTER_USER_KERNEL("dot").SetCreateFn<DotCpuKernel<dtype>>().SetIsMatchedHob(  \
      (user_op::HobDeviceTag() == device)                                          \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_DOT_CPU_KERNEL(DeviceType::kCPU, float)
REGISTER_DOT_CPU_KERNEL(DeviceType::kCPU, double)
REGISTER_DOT_CPU_KERNEL(DeviceType::kCPU, int32_t)
REGISTER_DOT_CPU_KERNEL(DeviceType::kCPU, int64_t)

#ifdef WITH_CUDA

template<typename T>
class DotGpuKernel final : public user_op::OpKernel {
 public:
  DotGpuKernel() = default;
  ~DotGpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    int64_t n = x->shape().elem_cnt();
    CHECK(n <= INT_MAX);
    NewKernelUtil<DeviceType::kGPU>::OFDot(ctx->device_ctx(), n, x->dptr<T>(), 1, y->dptr<T>(), 1, out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#endif

#ifdef WITH_CUDA
#define REGISTER_DOT_GPU_KERNEL(device, dtype)                                    \
  REGISTER_USER_KERNEL("dot").SetCreateFn<DotGpuKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                         \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_DOT_GPU_KERNEL(DeviceType::kGPU, float)
REGISTER_DOT_GPU_KERNEL(DeviceType::kGPU, double)

#endif

}  // namespace

}  // namespace oneflow
