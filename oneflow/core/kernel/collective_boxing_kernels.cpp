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
#include "oneflow/core/lazy/actor/collective_boxing_actor_context.h"

namespace oneflow {

using namespace boxing::collective;

namespace {

CollectiveBoxingActorContext* GetCollectiveBoxingActorContext(KernelContext* kernel_ctx) {
  auto* actor_context_provider = CHECK_NOTNULL(dynamic_cast<ActorContextProvider*>(kernel_ctx));
  return CHECK_NOTNULL(
      dynamic_cast<CollectiveBoxingActorContext*>(actor_context_provider->GetActorContext()));
}

class CollectiveBoxingKernelState final : public KernelState {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingKernelState);
  explicit CollectiveBoxingKernelState(const RankDesc& rank_desc)
      : request_handle_(Singleton<Scheduler>::Get()->CreateRequestHandle(rank_desc)) {}
  ~CollectiveBoxingKernelState() override {
    Singleton<Scheduler>::Get()->DestroyRequestHandle(request_handle_);
  }
  RequestHandle* request_handle() { return request_handle_; }

 private:
  RequestHandle* request_handle_ = nullptr;
};

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
  const void* send_buff = nullptr;
  void* recv_buff = nullptr;
  const RankDesc& rank_desc = this->op_conf().collective_boxing_generic_conf().rank_desc();
  const DataType data_type = rank_desc.op_desc().data_type();
  if (GenericOpHasInput(rank_desc)) {
    const Blob* in = ctx->BnInOp2Blob("in");
    CHECK_EQ(in->data_type(), data_type);
    CHECK(in->shape() == ShapeView(GenericOpGetInputShape(rank_desc)));
    send_buff = in->dptr();
  }
  if (GenericOpHasOutput(rank_desc)) {
    Blob* out = ctx->BnInOp2Blob("out");
    CHECK_EQ(out->data_type(), data_type);
    CHECK(out->shape() == ShapeView(GenericOpGetOutputShape(rank_desc)));
    recv_buff = out->mut_dptr();
  }
  auto* actor_ctx = GetCollectiveBoxingActorContext(ctx);
  actor_ctx->Schedule(request_handle, send_buff, recv_buff);
}

REGISTER_KERNEL(OperatorConf::kCollectiveBoxingGenericConf, CollectiveBoxingGenericKernel);

}  // namespace

}  // namespace oneflow
