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

template <typename T>
void reverse(T *y, int start, int end) {
    while(start < end) {
        std::swap(y[start], y[end]);
        start += 1;
        end -= 1;
    }
}

template <typename T>
void Roll(DeviceCtx *ctx, std::vector<int32_t> move, int64_t n, const T *x, T *y) {
    for(int64_t i = 0; i != n; ++i) {
        y[i] = x[i];
    }
    if(move.size() == 1) {
        int k = move[0];
        k %= n;
        reverse(y, 0, n-1);
        reverse(y, 0, k-1);
        reverse(y, k, n-1);
    }
}

template<DeviceType device_type, typename T>
class RollKernel final : public user_op::OpKernel {
public: 
    RollKernel() = default;
    ~RollKernel() override = default;

private:
    void Compute(user_op::KernelComputeContext* ctx) const override {
        const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
        user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
        const int64_t n = in->shape().elem_cnt();
        const std::vector<int32_t> move = ctx->Attr<std::vector<int32_t>>("shifts");
        Roll<T>(ctx->device_ctx(),
           move,
           n,
           in->dptr<T>(),
           out->mut_dptr<T>());
    }
    bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};
#define REGISTER_ROLL_KERNEL(device, dtype)                                                             \
  REGISTER_USER_KERNEL("roll")                                                                                     \
      .SetCreateFn<RollKernel<device, dtype>>()                                                         \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device));         \

REGISTER_ROLL_KERNEL(DeviceType::kCPU, float)
REGISTER_ROLL_KERNEL(DeviceType::kCPU, double)

}
}