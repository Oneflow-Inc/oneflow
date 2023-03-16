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
#include "oneflow/core/common/data_type.h"
#include "oneflow/core/cuda/elementwise.cuh"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/util/cuda_half_util.h"
#include "oneflow/core/ep/cuda/cuda_stream.h"
namespace oneflow {

namespace {

// Write ReLU Functor.
template<typename T>
struct FracForwardGpu {
  OF_DEVICE_FUNC T operator()(T x) const { return x - std::trunc(x); }
};

}  // namespace

template<typename T>
class GpuFracKernel final : public user_op::OpKernel {
 public:
  GpuFracKernel() = default;
  ~GpuFracKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const int32_t elem_cnt = x->shape_view().elem_cnt();
    // Use CUDA Elementwise Template.
    OF_CUDA_CHECK(
        (cuda::elementwise::Unary(FracForwardGpu<T>(), elem_cnt, y->mut_dptr<T>(), x->dptr<T>(),
                                  ctx->stream()->As<ep::CudaStream>()->cuda_stream())));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_GPU_FRAC_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("frac").SetCreateFn<GpuFracKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                               \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_GPU_FRAC_KERNEL(float)
REGISTER_GPU_FRAC_KERNEL(double)

}  // namespace oneflow