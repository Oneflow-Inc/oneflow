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
#include "oneflow/core/common/nd_index_offset_helper.h"

namespace oneflow {

namespace {
template<typename T>
__global__ void FillForwardGpu(const int n, const float value, T* y) {
  CUDA_1D_KERNEL_LOOP(i, n) { y[i] = value; }
}
};  // namespace

template<typename T>
class FillGpuKernel final : public user_op::OpKernel {
 public:
  FillGpuKernel() = default;
  ~FillGpuKernel() = default;

 private:
  using user_op::OpKernel::Compute;
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const float value = ctx->Attr<float>("value");
    const int32_t elem_cnt = x->shape().elem_cnt();
    T* y_ptr = y->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, elem_cnt) { y_ptr[i] = value; }
    RUN_CUDA_KERNEL((FillForwardGpu<T>), ctx->stream(), elem_cnt, elem_cnt, value,
                    y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FILL_CUDA_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("fill_").SetCreateFn<FillGpuKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCUDA)                                \
      && (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));

REGISTER_FILL_CUDA_KERNEL(bool)
REGISTER_FILL_CUDA_KERNEL(float)
REGISTER_FILL_CUDA_KERNEL(double)
REGISTER_FILL_CUDA_KERNEL(uint8_t)
REGISTER_FILL_CUDA_KERNEL(int8_t)
REGISTER_FILL_CUDA_KERNEL(int32_t)
REGISTER_FILL_CUDA_KERNEL(int64_t)

}  // namespace oneflow
