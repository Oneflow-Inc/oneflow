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

namespace oneflow {

template<typename T>
class CpuMedianWithIndicesKernel final : public user_op::OpKernel {
 public:
  CpuMedianWithIndicesKernel() = default;
  ~CpuMedianWithIndicesKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("input", 0);
    const int64_t num_axes = in->shape_view().NumAxes();
    const int64_t size = in->shape_view().elem_cnt();
    if (size == 0) return;
    const int64_t stride = in->shape_view().At(num_axes - 1);
    const int64_t instance_num = size / stride;
    user_op::Tensor* values = ctx->Tensor4ArgNameAndIndex("values", 0);
    user_op::Tensor* indices = ctx->Tensor4ArgNameAndIndex("indices", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);
    Memcpy<DeviceType::kCPU>(ctx->stream(), tmp_buffer->mut_dptr<void>(), in->dptr<void>(),
                             size * sizeof(T));
    const int64_t thread_num =
        std::min(instance_num, (int64_t)Singleton<ThreadPool>::Get()->thread_num());
    const BalancedSplitter bs(instance_num, thread_num);
    BlockingCounter bc(thread_num);
    FOR_RANGE(int64_t, thread_id, 0, thread_num) {
      const Range range = bs.At(thread_id);
      Singleton<ThreadPool>::Get()->AddWork([=, &bc]() {
        FOR_RANGE(int64_t, i, range.begin(), range.end()) {
          T* in_ptr = tmp_buffer->mut_dptr<T>() + i * stride;
          T* val_ptr = values->mut_dptr<T>() + i;
          int64_t* ind_ptr = indices->mut_dptr<int64_t>() + i;
          std::vector<int64_t> idx(stride);
          auto first = idx.begin();
          auto last = idx.end();
          std::iota(first, last, 0);
          auto nth = first;
          nth += (stride - 1) / 2;
          std::nth_element(first, nth, last, [&in_ptr](int64_t i, int64_t j) {
            return in_ptr[i] < in_ptr[j] || (in_ptr[i] == in_ptr[j] && i < j);
          });
          *val_ptr = in_ptr[*nth];
          *ind_ptr = *nth;
        }
        bc.Decrease();
      });
    }
    bc.WaitForeverUntilCntEqualZero();
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_MEDIAN_WITH_INDICES_KERNEL(dtype)                                     \
  REGISTER_USER_KERNEL("median_with_indices")                                              \
      .SetCreateFn<CpuMedianWithIndicesKernel<dtype>>()                                    \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                      \
                       && (user_op::HobDataType("input", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) -> size_t {                        \
        return ctx->InputShape("input", 0).elem_cnt() * sizeof(dtype);                     \
      });

REGISTER_CPU_MEDIAN_WITH_INDICES_KERNEL(float)
REGISTER_CPU_MEDIAN_WITH_INDICES_KERNEL(double)
REGISTER_CPU_MEDIAN_WITH_INDICES_KERNEL(int8_t)
REGISTER_CPU_MEDIAN_WITH_INDICES_KERNEL(uint8_t)
REGISTER_CPU_MEDIAN_WITH_INDICES_KERNEL(int32_t)
REGISTER_CPU_MEDIAN_WITH_INDICES_KERNEL(int64_t)

}  // namespace oneflow
