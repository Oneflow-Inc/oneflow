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

template<typename T>
class CpuCumsumKernel final : public user_op::OpKernel {
 public:
  CpuCumsumKernel() = default;
  ~CpuCumsumKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);

    Memcpy<DeviceType::kCPU>(ctx->device_ctx(), out->mut_dptr<T>(), in->dptr<T>(),
                             in->shape().elem_cnt() * sizeof(T));
    int32_t axis = ctx->Attr<int32_t>("axis");
    const bool exclusive = ctx->Attr<bool>("exclusive");
    const bool reverse = ctx->Attr<bool>("reverse");
    const int32_t elem_cnt = in->shape().elem_cnt();
    const int32_t instance_size = in->shape().At(axis);
    const int32_t instance_num = elem_cnt / instance_size;
    int32_t post = 1;
    FOR_RANGE(int32_t, i, axis + 1, in->shape().NumAxes()) { post *= in->shape().At(i); }
    const T* in_ptr = in->dptr<T>();
    T* out_ptr = out->mut_dptr<T>();
    FOR_RANGE(int32_t, i, 0, instance_num) {
      const int32_t start_idx = reverse ? i % post + (i / post + 1) * instance_size * post - post
                                        : i % post + (i / post) * instance_size * post;
      out_ptr[start_idx] = 0;
      T temp = 0;
      FOR_RANGE(int32_t, j, exclusive, instance_size) {
        int32_t out_index = reverse ? start_idx - j * post : start_idx + j * post;
        int32_t in_index =
            reverse ? start_idx - (j - exclusive) * post : start_idx + (j - exclusive) * post;
        temp += in_ptr[in_index];
        out_ptr[out_index] = temp;
      }
    }
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CPU_CUMSUM_KERNEL(dtype)                                               \
  REGISTER_USER_KERNEL("cumsum").SetCreateFn<CpuCumsumKernel<dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == "cpu")                                                \
      & (user_op::HobDataType("out", 0) == GetDataType<dtype>::value));

REGISTER_CPU_CUMSUM_KERNEL(float)
REGISTER_CPU_CUMSUM_KERNEL(double)
REGISTER_CPU_CUMSUM_KERNEL(int32_t)
REGISTER_CPU_CUMSUM_KERNEL(int64_t)
}  // namespace oneflow 
