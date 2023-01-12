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
#include "oneflow/core/kernel/kernel_util.h"
#include "oneflow/core/thread/thread_manager.h"
#include <random>

namespace oneflow {

template<typename T>
class CpuRReluKernel final : public user_op::OpKernel {
 public:
  CpuRReluKernel() = default;
  ~CpuRReluKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const int64_t size = in->shape_view().elem_cnt();
    if (size == 0) return;

    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("output", 0);
    user_op::Tensor* noise_data = ctx->Tensor4ArgNameAndIndex("noise_data", 0);
    const T& lower = ctx->Attr<float>("lower");
    const T& upper = ctx->Attr<float>("upper");

    T* out_ptr = out->mut_dptr<T>();
    T* noise_ptr = noise_data->mut_dptr<T>();
    const T* in_ptr = in->dptr<T>();

    const int64_t thread_num = (int64_t)Singleton<ThreadPool>::Get()->thread_num();
    const BalancedSplitter bs(size, thread_num);
    BlockingCounter bc(thread_num);

    std::mt19937 gen{std::random_device{}()};
    std::uniform_real_distribution<> uni_random_float(lower, upper);
    FOR_RANGE(int64_t, thread_id, 0, thread_num) {
      const Range range = bs.At(thread_id);
      Singleton<ThreadPool>::Get()->AddWork([=, &bc, &gen, &uni_random_float]() {
        FOR_RANGE(int64_t, i, range.begin(), range.end()) {
          if (*(in_ptr + i) >= 0) {
            noise_ptr[i] = 1;
            out_ptr[i] = in_ptr[i];
          } else {
            T random_data = uni_random_float(gen);
            noise_ptr[i] = random_data;
            out_ptr[i] = in_ptr[i] * random_data;
          }
        }
        bc.Decrease();
      });
    }
    bc.WaitForeverUntilCntEqualZero();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_RRelu_KERNEL(dtype)                                              \
  REGISTER_USER_KERNEL("rrelu").SetCreateFn<CpuRReluKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == DeviceType::kCPU)                                  \
      && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CPU_RRelu_KERNEL(float) 
REGISTER_CPU_RRelu_KERNEL(double)

}  // namespace oneflow
