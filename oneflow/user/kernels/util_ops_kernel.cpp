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
#include "oneflow/core/common/device_type.pb.h"
#include "oneflow/core/common/data_type_seq.h"
#include "oneflow/core/framework/framework.h"
#include "oneflow/core/framework/op_kernel.h"
#include "oneflow/user/kernels/util_ops_kernel_functor.h"

namespace oneflow {
namespace user_op {

template<DeviceType kernel_device, typename T,
         template<DeviceType functor_device, typename U> class UtilOpsFunctor>
class UtilOpsKernel final : public OpKernel {
 public:
  UtilOpsKernel() = default;
  ~UtilOpsKernel() = default;

 private:
  void Compute(KernelComputeContext* ctx) const override {
    const auto* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    auto* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    const auto x_elem_cnt = x->shape().elem_cnt();
    const auto y_elem_cnt = y->shape().elem_cnt();
    CHECK_EQ(x_elem_cnt, y_elem_cnt);
    const T* x_ptr = x->dptr<T>();
    bool* y_ptr = y->mut_dptr<bool>();
    UtilOpsFunctor<kernel_device, T>()(ctx->stream(), y_ptr, x_ptr, x_elem_cnt);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_UTIL_OPS_KERNEL(device, dtype)                                              \
  REGISTER_USER_KERNEL("isnan")                                                              \
      .SetCreateFn<UtilOpsKernel<device, OF_PP_PAIR_FIRST(dtype), IsNanFunctor>>()           \
      .SetIsMatchedHob(                                                                      \
          (user_op::HobDeviceType() == device)                                               \
          && (user_op::HobDataType("x", 0) == GetDataType<OF_PP_PAIR_FIRST(dtype)>::value)); \
  REGISTER_USER_KERNEL("isinf")                                                              \
      .SetCreateFn<UtilOpsKernel<device, OF_PP_PAIR_FIRST(dtype), IsInfFunctor>>()           \
      .SetIsMatchedHob(                                                                      \
          (user_op::HobDeviceType() == device)                                               \
          && (user_op::HobDataType("x", 0) == GetDataType<OF_PP_PAIR_FIRST(dtype)>::value));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UTIL_OPS_KERNEL, (DeviceType::kCPU),
                                 UTIL_OPS_FUNCTOR_DTYPE_SEQ);
OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_UTIL_OPS_KERNEL, (DeviceType::kCUDA),
                                 UTIL_OPS_FUNCTOR_DTYPE_SEQ);
#undef REGISTER_UTIL_OPS_KERNEL

}  // namespace user_op
}  // namespace oneflow
