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

class ConstState final : public OpKernelState {
 public:
  ConstState(bool is_init) : is_init_(is_init) {}
  ~ConstState() = default;
  bool is_inited() const { return is_init_; }
  void set_is_inited(bool val) { is_init_ = val; }

 private:
  bool is_init_;
};

template<DeviceType device_type, typename T>
class ConstantKernel final : public OpKernel {
 public:
  ConstantKernel() = default;
  ~ConstantKernel() = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<ConstState>(false);
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    auto* const_state = dynamic_cast<ConstState*>(state);
    if (const_state->is_inited()) { return; }
    Tensor* out_tensor = ctx->Tensor4ArgNameAndIndex("out", 0);
    bool is_floating_value = ctx->Attr<bool>("is_floating_value");
    const int64_t elem_cnt = out_tensor->shape().elem_cnt();
    CHECK_GT(elem_cnt, 0);
    NewKernelUtil<device_type>::Fill(ctx->device_ctx(), elem_cnt,
                                     is_floating_value
                                         ? static_cast<T>(ctx->Attr<double>("floating_value"))
                                         : static_cast<T>(ctx->Attr<int64_t>("integer_value")),
                                     out_tensor->mut_dptr<T>());

    const_state->set_is_inited(true);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_CONSTANT_XPU_KERNEL(device, dtype)        \
  REGISTER_USER_KERNEL("constant")                         \
      .SetCreateFn<ConstantKernel<device, dtype>>()        \
      .SetIsMatchedHob((user_op::HobDeviceTag() == device) \
                       & (user_op::HobAttr<DataType>("dtype") == GetDataType<dtype>::value));

#define REGISTER_CONSTANT_KERNEL(device, dtype_pair) \
  REGISTER_CONSTANT_XPU_KERNEL(device, OF_PP_PAIR_FIRST(dtype_pair))

OF_PP_SEQ_PRODUCT_FOR_EACH_TUPLE(REGISTER_CONSTANT_KERNEL, DEVICE_TYPE_SEQ,
                                 ARITHMETIC_DATA_TYPE_SEQ)

}  // namespace user_op
}  // namespace oneflow
