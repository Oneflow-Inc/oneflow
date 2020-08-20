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
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

template<typename T>
class CpuArgMaxKernel final : public user_op::OpKernel {
 public:
  CpuArgMaxKernel() = default;
  ~CpuArgMaxKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const T* in_ptr = in->dptr<T>();
    int32_t* out_ptr = out->mut_dptr<int32_t>();

    const int32_t instance_size = in->shape().At(in->shape().NumAxes() - 1);
    const int32_t instance_num = in->shape().elem_cnt() / instance_size;
    const int32_t num_thread = std::min(instance_num, Global<ThreadPool>::Get()->thread_num());
    const BalancedSplitter bs(instance_num, num_thread);
    BlockingCounter bc(num_thread);
    FOR_RANGE(int32_t, thread_id, 0, num_thread) {
      const Range range = bs.At(thread_id);
      Global<ThreadPool>::Get()->AddWork([=, &bc]() {
        FOR_RANGE(int32_t, i, range.begin(), range.end()) {
          const T* in_ptr_i = in_ptr + i * instance_size;
          out_ptr[i] =
              std::distance(in_ptr_i, std::max_element(in_ptr_i, in_ptr_i + instance_size));
        }
        bc.Decrease();
      });
    }
    bc.WaitUntilCntEqualZero();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_ARGMAX_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("argmax").SetCreateFn<CpuArgMaxKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "cpu")                                                \
      & (user_op::HobDataType("in", 0) == GetDataType<dtype>::value));

REGISTER_CPU_ARGMAX_KERNEL(float)
REGISTER_CPU_ARGMAX_KERNEL(double)
REGISTER_CPU_ARGMAX_KERNEL(int32_t)
REGISTER_CPU_ARGMAX_KERNEL(int64_t)

}  // namespace oneflow
