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
#include "oneflow/user/kernels/op_kernel_state_wrapper.h"

namespace oneflow {

namespace {

template<DeviceType device_type>
class PackKernel final : public user_op::OpKernel {
 public:
  PackKernel() = default;
  ~PackKernel() override = default;

  std::shared_ptr<user_op::OpKernelState> CreateOpKernelState(
      user_op::KernelInitContext* ctx) const override {
    return std::make_shared<OpKernelStateWrapper<std::pair<size_t, size_t>>>(
        std::make_pair<size_t, size_t>(0, 0));
  }

 private:
  void Compute(user_op::KernelComputeContext* ctx, user_op::OpKernelState* state) const override {
    const user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    CHECK_GT(in->shape().NumAxes(), 0);
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    CHECK_EQ(in->data_type(), out->data_type());
    CHECK_EQ(in->shape().NumAxes(), out->shape().NumAxes());
    const auto pack_num = ctx->Attr<int32_t>("pack_num");
    CHECK_EQ(out->shape().At(0), in->shape().At(0) * pack_num);
    for (int64_t i = 1; i < in->shape().NumAxes(); ++i) {
      CHECK_EQ(out->shape().At(i), in->shape().At(i));
    }
    const int64_t copy_size = in->shape().elem_cnt() * GetSizeOfDataType(out->data_type());
    auto* state_wrapper = dynamic_cast<OpKernelStateWrapper<std::pair<size_t, size_t>>*>(state);
    CHECK_NOTNULL(state_wrapper);
    const size_t index = state_wrapper->Get().first;
    CHECK_EQ(state_wrapper->Get().second, pack_num);
    Memcpy<device_type>(ctx->device_ctx(), out->mut_dptr<char>() + index * copy_size,
                        in->dptr<char>(), copy_size);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_PACK_KERNEL(device)                                              \
  REGISTER_USER_KERNEL("pack").SetCreateFn<PackKernel<device>>().SetIsMatchedHob( \
      (user_op::HobDeviceTag() == device));

OF_PP_FOR_EACH_TUPLE(REGISTER_PACK_KERNEL, DEVICE_TYPE_SEQ)

#undef REGISTER_PACK_KERNEL

}  // namespace

}  // namespace oneflow
