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
#include "oneflow/user/kernels/where_kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename T, typename CondT>
class WhereKernel final : public user_op::OpKernel {
 public:
  WhereKernel() = default;
  ~WhereKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    const user_op::Tensor* cond = ctx->Tensor4ArgNameAndIndex("condition", 0);
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    const user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    WhereKernelUtil<device_type, T, CondT>::Where(ctx->device_ctx(), out->shape().elem_cnt(),
                                                  cond->dptr<CondT>(), x->dptr<T>(), y->dptr<T>(),
                                                  out->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_WHERE_KERNEL(device_type_v, dtype_pair, ctype_pair)                             \
  REGISTER_USER_KERNEL("where")                                                                  \
      .SetCreateFn<WhereKernel<device_type_v, OF_PP_PAIR_FIRST(dtype_pair),                      \
                               OF_PP_PAIR_FIRST(ctype_pair)>>()                                  \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device_type_v)                                \
                       & (user_op::HobDataType("condition", 0) == OF_PP_PAIR_SECOND(ctype_pair)) \
                       & (user_op::HobDataType("out", 0) == OF_PP_PAIR_SECOND(dtype_pair)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_WHERE_KERNEL, DEVICE_TYPE_SEQ, ARITHMETIC_DATA_TYPE_SEQ,
                                 INT_DATA_TYPE_SEQ)

}  // namespace oneflow
