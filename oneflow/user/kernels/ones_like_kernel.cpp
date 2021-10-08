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
#include "oneflow/core/common/switch_func.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/kernel/new_kernel_util.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename T>
void FillOnes(DeviceCtx* device_ctx, user_op::Tensor* output) {
  NewKernelUtil<device_type>::Fill(device_ctx, output->shape().elem_cnt(), static_cast<T>(1),
                                   output->mut_dptr<T>());
}

template<DeviceType device_type>
struct FillOnesUtil {
#define MAKE_FILL_ONES_SWITCH_ENTRY(func_name, T) func_name<device_type, T>
  DEFINE_STATIC_SWITCH_FUNC(
      void, FillOnes, MAKE_FILL_ONES_SWITCH_ENTRY,
      MAKE_DATA_TYPE_CTRV_SEQ(ARITHMETIC_DATA_TYPE_SEQ UNSIGNED_INT_DATA_TYPE_SEQ));
#undef MAKE_FILL_ONES_SWITCH_ENTRY
};

}  // namespace

template<DeviceType device_type>
class OnesLikeKernel final : public user_op::OpKernel {
 public:
  OnesLikeKernel() = default;
  ~OnesLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& data_type = out->data_type();
    FillOnesUtil<device_type>::SwitchFillOnes(SwitchCase(data_type), ctx->device_ctx(), out);
  }

  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_ONES_LIKE_KERNEL(device_type_v)    \
  REGISTER_USER_KERNEL("ones_like")                 \
      .SetCreateFn<OnesLikeKernel<device_type_v>>() \
      .SetIsMatchedHob(user_op::HobDeviceTag() == device_type_v);

REGISTER_ONES_LIKE_KERNEL(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_ONES_LIKE_KERNEL(DeviceType::kGPU)
#endif

}  // namespace oneflow
