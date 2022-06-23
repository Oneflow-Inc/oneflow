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
#include "oneflow/core/ccl/ccl.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/framework/instructions_builder.h"

namespace oneflow {

namespace {

template<DeviceType device_type>
class SendKernel final : public user_op::OpKernel {
 public:
  SendKernel() = default;
  ~SendKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto& dst_process_id = ctx->Attr<int64_t>("dst_process_id");
    CHECK_JUST(ccl::Send<device_type>(in->dptr(), in->shape_view().elem_cnt(), in->data_type(),
                                      dst_process_id, ctx->stream()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

template<DeviceType device_type>
class RecvKernel final : public user_op::OpKernel {
 public:
  RecvKernel() = default;
  ~RecvKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& src_process_id = ctx->Attr<int64_t>("src_process_id");
    CHECK_JUST(ccl::Recv<device_type>(out->mut_dptr(), out->shape_view().elem_cnt(),
                                      out->data_type(), src_process_id, ctx->stream()));
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

#define REGISTER_KERNEL(device)                                                   \
  REGISTER_USER_KERNEL("send").SetCreateFn<SendKernel<device>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device));                                      \
  REGISTER_USER_KERNEL("recv").SetCreateFn<RecvKernel<device>>().SetIsMatchedHob( \
      (user_op::HobDeviceType() == device));

REGISTER_KERNEL(DeviceType::kCPU)
#ifdef WITH_CUDA
REGISTER_KERNEL(DeviceType::kCUDA)
#endif
}  // namespace

}  // namespace oneflow
