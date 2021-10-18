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
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/job/collective_boxing/scheduler.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/graph/boxing/collective_boxing_util.h"
#include "oneflow/core/device/collective_boxing_device_context.h"

namespace oneflow {

class CollectiveBoxingKernelState final : public KernelState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingKernelState);
  explicit CollectiveBoxingKernelState(const RankDesc& rank_desc)
      : request_handle_(Global<Scheduler>::Get()->CreateRequestHandle(rank_desc)) {}
  ~CollectiveBoxingKernelState() override {
    Global<Scheduler>::Get()->DestroyRequestHandle(request_handle_);
  }
  RequestHandle* request_handle() { return request_handle_; }

 private:
  RequestHandle* request_handle_ = nullptr;
};

using namespace boxing::collective;

class CollectiveBoxingGenericKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingGenericKernel);
  CollectiveBoxingGenericKernel() = default;
  ~CollectiveBoxingGenericKernel() override = default;

 private:
  void VirtualKernelInit(KernelContext* ctx) override;
  bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
};

void CollectiveBoxingGenericKernel::VirtualKernelInit(KernelContext* ctx) {
  const RankDesc& rank_desc = this->op_conf().collective_boxing_generic_conf().rank_desc();
  ctx->set_state(std::make_shared<CollectiveBoxingKernelState>(rank_desc));
}

void CollectiveBoxingGenericKernel::ForwardDataContent(KernelContext* ctx) const {
  RequestHandle* request_handle =
      CHECK_NOTNULL(dynamic_cast<CollectiveBoxingKernelState*>(ctx->state().get()))
          ->request_handle();
  auto request = std::make_shared<RuntimeRequestInfo>();
  const RankDesc& rank_desc = this->op_conf().collective_boxing_generic_conf().rank_desc();
  const DataType data_type = rank_desc.op_desc().data_type();
  if (GenericOpHasInput(rank_desc)) {
    const Blob* in = ctx->BnInOp2Blob("in");
    CHECK_EQ(in->data_type(), data_type);
    CHECK(in->shape() == ShapeView(GenericOpGetInputShape(rank_desc)));
    request->send_buff = in->dptr();
  } else {
    request->send_buff = nullptr;
  }
  if (GenericOpHasOutput(rank_desc)) {
    Blob* out = ctx->BnInOp2Blob("out");
    CHECK_EQ(out->data_type(), data_type);
    CHECK(out->shape() == ShapeView(GenericOpGetOutputShape(rank_desc)));
    request->recv_buff = out->mut_dptr();
  } else {
    request->recv_buff = nullptr;
  }
  auto* device_ctx = dynamic_cast<CollectiveBoxingDeviceCtx*>(ctx->device_ctx());
  CHECK_NOTNULL(device_ctx);
  std::shared_ptr<CollectiveBoxingDeviceCtxCheckpoint> checkpoint = device_ctx->AddCheckpoint();
  request->callback = [checkpoint](const Maybe<void>& status) {
    CHECK(status.IsOk());
    checkpoint->SetDone();
  };
  Global<Scheduler>::Get()->Schedule(request_handle, request);
}

REGISTER_KERNEL(OperatorConf::kCollectiveBoxingGenericConf, CollectiveBoxingGenericKernel);

}  // namespace oneflow
