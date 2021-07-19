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
#include "oneflow/user/kernels/upsample_kernel.h"

namespace oneflow {

template<typename T>
class FlipCPUKernel final : public user_op::OpKernel {
 public:
  FlipCPUKernel() = default;
  ~FlipCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* x_tensor = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y_tensor = ctx->Tensor4ArgNameAndIndex("y", 0);
    const T* x_ptr = x_tensor->dptr<T>();
    T* y_ptr = y_tensor->mut_dptr<T>();
    std::vector<int32_t> dims = ctx->Attr<std::vector<int32_t>>("dims");
    std::unordered_map<int, bool> mp;
    for (auto x : dims) { mp[x] = true; }
    const int32_t out_dims = y_tensor->shape().NumAxes();
    std::vector<int32_t> dim_sum(out_dims + 1);
    dim_sum[out_dims] = 1;
    for (int i = out_dims - 1; i >= 0; i--) {
      dim_sum[i] = dim_sum[i + 1] * y_tensor->shape().At(i);
    }
    for (int i = 0; i < out_dims; i++) {
      int offset = i * dim_sum[i + 1];
      int dim_len = y_tensor->shape().At(i);
      for (int j = 0; j < dim_len; j++) {
        if (mp[i]) {
          *(y_ptr + offset + (dim_len - j - 1)) = *(x_ptr + offset + j);
        } else {
          *(y_ptr + offset + j) = *(x_ptr + offset + j);
        }
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<typename T>
class FlipGrad1DCPUKernel final : public user_op::OpKernel {
 public:
  FlipGrad1DCPUKernel() = default;
  ~FlipGrad1DCPUKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* dx_tensor = ctx->Tensor4ArgNameAndIndex("dx", 0);
    Memset<DeviceType::kCPU>(ctx->device_ctx(), dx_tensor->mut_dptr<T>(), 0,
                             dx_tensor->shape().elem_cnt() * sizeof(T));
    const user_op::Tensor* dy_tensor = ctx->Tensor4ArgNameAndIndex("dy", 0);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_FLIP_CPU_KERNEL(dtype)                                             \
  REGISTER_USER_KERNEL("flip").SetCreateFn<FlipCPUKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "cpu")                                            \
      & (user_op::HobDataType("y", 0) == GetDataType<dtype>::value));               \
  REGISTER_USER_KERNEL("flip_grad")                                                 \
      .SetCreateFn<FlipGrad1DCPUKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceTag() == "cpu")                           \
                       & (user_op::HobDataType("dx", 0) == GetDataType<dtype>::value));

REGISTER_FLIP_CPU_KERNEL(float)
REGISTER_FLIP_CPU_KERNEL(double)
REGISTER_FLIP_CPU_KERNEL(int)

}  // namespace oneflow
