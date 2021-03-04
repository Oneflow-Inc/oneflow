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
#include "oneflow/core/kernel/square_sum_kernel_util.h"
#include "oneflow/core/common/balanced_splitter.h"

namespace oneflow {

namespace user_op {

template<DeviceType device_type, typename T>
class SquareSumKernel final : public user_op::OpKernel {
 public:
  SquareSumKernel() = default;
  ~SquareSumKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", 0);
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);

    SquareSumKernelUtil<device_type, T>::SquareSum(ctx->device_ctx(), x->shape().elem_cnt(),
                                                   x->dptr<T>(), y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_SQUARE_SUM_KERNEL(device, dtype)                      \
  REGISTER_USER_KERNEL("square_sum")                                   \
      .SetCreateFn<SquareSumKernel<device, OF_PP_PAIR_FIRST(dtype)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)             \
                       & (user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_SQUARE_SUM_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

template<DeviceType device_type, typename T>
class MultiSquareSumKernel final : public user_op::OpKernel {
 public:
  MultiSquareSumKernel() = default;
  ~MultiSquareSumKernel() override = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    std::vector<SquareSumParam<T>> params;
    params.resize(ctx->user_op_conf().input_size("x"));
    for (int64_t i = 0; i < params.size(); ++i) {
      const user_op::Tensor* x = ctx->Tensor4ArgNameAndIndex("x", i);
      params[i].count = x->shape().elem_cnt();
      params[i].ptr = x->dptr<T>();
    }
    user_op::Tensor* y = ctx->Tensor4ArgNameAndIndex("y", 0);
    SquareSumKernelUtil<device_type, T>::MultiSquareSum(ctx->device_ctx(), params,
                                                        y->mut_dptr<T>());
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_MULTI_SQUARE_SUM_KERNEL(device, dtype)                     \
  REGISTER_USER_KERNEL("multi_square_sum")                                  \
      .SetCreateFn<MultiSquareSumKernel<device, OF_PP_PAIR_FIRST(dtype)>>() \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device)                  \
                       & (user_op::HobDataType("y", 0) == OF_PP_PAIR_SECOND(dtype)));

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_MULTI_SQUARE_SUM_KERNEL, DEVICE_TYPE_SEQ,
                                 FLOATING_DATA_TYPE_SEQ)

}  // namespace user_op

}  // namespace oneflow
