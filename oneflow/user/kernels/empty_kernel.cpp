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
namespace user_op {

template<DeviceType device_type, typename T>
class EmptyKernel final : public OpKernel {
 public:
  EmptyKernel() = default;
  ~EmptyKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    const int64_t elem_cnt = out_tensor->shape().elem_cnt();
    CHECK_GT(elem_cnt, 0);

    // Do nothing
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_EMPTY_XPU_KERNEL(device, dtype)                                           \
  REGISTER_USER_KERNEL("empty").SetCreateFn<EmptyKernel<device, dtype>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device)                                                  \
      & (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

#define REGISTER_EMPTY_KERNEL(device, dtype_pair) \
  REGISTER_EMPTY_XPU_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_EMPTY_KERNEL, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ)

#ifdef WITH_CUDA
REGISTER_EMPTY_XPU_KERNEL(DeviceType::kGPU, float16);
#endif  // WITH_CUDA

}  // namespace user_op
}  // namespace oneflow
