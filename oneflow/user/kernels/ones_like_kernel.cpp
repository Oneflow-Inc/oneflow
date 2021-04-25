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

template<DeviceType device_type>
class OnesLikeKernel final : public user_op::OpKernel {
 public:
  OnesLikeKernel() = default;
  ~OnesLikeKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& dtype = out->data_type();
    switch (dtype) {
#define FILL_TENSOR_DATA_TYPE_CASE(cpp_type, data_type)                                    \
  case data_type: {                                                                        \
    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), out->shape().elem_cnt(),           \
                                     static_cast<cpp_type>(1), out->mut_dptr<cpp_type>()); \
    break;                                                                                 \
  }
      OF_PP_FOR_EACH_TUPLE(FILL_TENSOR_DATA_TYPE_CASE, FLOATING_DATA_TYPE_SEQ INT_DATA_TYPE_SEQ)
      default: {
        LOG(FATAL) << "Does not support data type " << dtype << " in OnesLikeKernel::Compute.";
      }
#undef FILL_TENSOR_DATA_TYPE_CASE
    }
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
