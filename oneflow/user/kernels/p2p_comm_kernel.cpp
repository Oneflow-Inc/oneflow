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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/rank_group.h"
#include "oneflow/core/framework/instructions_builder.h"
#include "oneflow/user/kernels/collective_communication/include/send.h"
#include "oneflow/user/kernels/collective_communication/include/recv.h"

namespace oneflow {

namespace {

namespace {

auto SendCollectiveCommunicationExists() {
  return hob::make_custom("SendCollectiveCommunicationExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            DeviceType device_type = ctx.device_type();
                            return ccl::IsSendRegistered(device_type);
                          });
}

auto RecvCollectiveCommunicationExists() {
  return hob::make_custom("RecvCollectiveCommunicationExists",
                          [=](const user_op::KernelRegContext& ctx) {
                            DeviceType device_type = ctx.device_type();
                            return ccl::IsRecvRegistered(device_type);
                          });
}

}  // namespace

class SendKernel final : public user_op::OpKernel {
 public:
  SendKernel() = default;
  ~SendKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* in = ctx->Tensor4ArgNameAndIndex("in", 0);
    const auto& dst_process_id = ctx->Attr<int64_t>("dst_process_id");
    std::unique_ptr<ccl::Send> send =
        ccl::NewCollectiveCommunication<ccl::Send>(ctx->device_type(), in->data_type());
    send->Launch(ctx->stream(), in->dptr(), in->shape_view().elem_cnt(), dst_process_id);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

class RecvKernel final : public user_op::OpKernel {
 public:
  RecvKernel() = default;
  ~RecvKernel() = default;

 private:
  void Compute(user_op::KernelComputeContext* ctx) const override {
    user_op::Tensor* out = ctx->Tensor4ArgNameAndIndex("out", 0);
    const auto& src_process_id = ctx->Attr<int64_t>("src_process_id");
    std::unique_ptr<ccl::Recv> recv =
        ccl::NewCollectiveCommunication<ccl::Recv>(ctx->device_type(), out->data_type());
    recv->Launch(ctx->stream(), out->mut_dptr(), out->shape_view().elem_cnt(), src_process_id);
  }
  bool AlwaysComputeWhenAllOutputsEmpty() const override { return false; }
};

REGISTER_USER_KERNEL("send").SetCreateFn<SendKernel>().SetIsMatchedHob(
    SendCollectiveCommunicationExists());

REGISTER_USER_KERNEL("recv").SetCreateFn<RecvKernel>().SetIsMatchedHob(
    RecvCollectiveCommunicationExists());
}  // namespace

}  // namespace oneflow
