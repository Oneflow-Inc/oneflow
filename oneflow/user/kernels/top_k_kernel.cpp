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

namespace {

template<typename T>
void ComputeTopOne(const T* in_ptr, const Range& range, int64_t instance_size, int64_t* out_ptr) {
  FOR_RANGE(int64_t, i, range.begin(), range.end()) {
    const T* in_ptr_i = in_ptr + i * instance_size;
    out_ptr[i] = std::distance(in_ptr_i, std::max_element(in_ptr_i, in_ptr_i + instance_size));
  }
}

template<typename T>
void ComputeTopK(const T* in_ptr, int64_t* indices_ptr, const Range& range, int64_t instance_size,
                 int64_t k, bool sorted, int64_t* out_ptr) {
  FOR_RANGE(int64_t, i, range.begin(), range.end()) {
    const int64_t offset = i * instance_size;
    const T* in_ptr_i = in_ptr + offset;
    int64_t* indices_ptr_i = indices_ptr + offset;
    std::iota(indices_ptr_i, indices_ptr_i + instance_size, 0);
    auto comp = [&](const int64_t lhs, const int64_t rhs) {
      const T l = in_ptr_i[lhs];
      const T r = in_ptr_i[rhs];
      if (l == r) {
        return lhs < rhs;
      } else {
        return l > r;
      }
    };
    std::nth_element(indices_ptr_i, indices_ptr_i + k, indices_ptr_i + instance_size, comp);
    if (sorted) { std::sort(indices_ptr_i, indices_ptr_i + k, comp); }
    std::copy(indices_ptr_i, indices_ptr_i + k, out_ptr + i * k);
  }
}

template<typename T>
void CpuTopK(ep::Stream* /*stream*/, const T* in_ptr, int64_t* indices_ptr, int64_t instance_num,
             int64_t instance_size, int64_t k, bool sorted, int64_t* out_ptr) {
  const int64_t num_thread =
      std::min(instance_num, static_cast<int64_t>(Singleton<ThreadPool>::Get()->thread_num()));
  const BalancedSplitter bs(instance_num, num_thread);
  BlockingCounter bc(num_thread);
  FOR_RANGE(int64_t, thread_id, 0, num_thread) {
    const Range range = bs.At(thread_id);
    Singleton<ThreadPool>::Get()->AddWork([=, &bc]() {
      if (k == 1) {
        ComputeTopOne(in_ptr, range, instance_size, out_ptr);
      } else {
        ComputeTopK(in_ptr, indices_ptr, range, instance_size, k, sorted, out_ptr);
      }
      bc.Decrease();
    });
  }
  bc.WaitForeverUntilCntEqualZero();
}

}  // namespace

template<typename T>
class TopKCpuKernel final : public user_op::OpKernel {
 public:
  TopKCpuKernel() = default;
  ~TopKCpuKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    if (in->shape_view().elem_cnt() == 0) { return; }
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    user_op::Tensor* tmp_buffer = ctx->Tensor4ArgNameAndIndex("tmp_buffer", 0);

    const int64_t instance_size = in->shape_view().At(in->shape_view().NumAxes() - 1);
    const int64_t instance_num = in->shape_view().elem_cnt() / instance_size;
    const int64_t k = std::min(static_cast<int64_t>(ctx->Attr<int32_t>("k")), instance_size);
    int64_t* indices_ptr = tmp_buffer ? tmp_buffer->mut_dptr<int64_t>() : nullptr;
    CpuTopK(ctx->stream(), in->dptr<T>(), indices_ptr, instance_num, instance_size, k,
            ctx->Attr<bool>("sorted"), out->mut_dptr<int64_t>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_TOP_K_KERNEL(dtype)                                                \
  REGISTER_USER_KERNEL("top_k")                                                         \
      .SetCreateFn<TopKCpuKernel<dtype>>()                                              \
      .SetIsMatchedHob((user_op::HobDeviceType() == DeviceType::kCPU)                   \
                       && (user_op::HobDataType("in", 0) == GetDataType<dtype>::value)) \
      .SetInferTmpSizeFn([](user_op::InferContext* ctx) {                               \
        const Shape& in_shape = ctx->InputShape("in", 0);                               \
        return ctx->Attr<int32_t>("k") > 1 ? in_shape.elem_cnt() * sizeof(int64_t) : 0; \
      });

REGISTER_CPU_TOP_K_KERNEL(float)
REGISTER_CPU_TOP_K_KERNEL(double)
REGISTER_CPU_TOP_K_KERNEL(int8_t)
REGISTER_CPU_TOP_K_KERNEL(uint8_t)
REGISTER_CPU_TOP_K_KERNEL(int32_t)
REGISTER_CPU_TOP_K_KERNEL(int64_t)

}  // namespace oneflow
