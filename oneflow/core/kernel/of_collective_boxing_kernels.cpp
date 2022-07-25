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
#include "oneflow/core/job/of_collective_boxing/collective_manager.h"
#include "oneflow/core/common/blocking_counter.h"
#include "oneflow/core/graph/boxing/of_collective_boxing_util.h"
#include "oneflow/core/lazy/actor/of_collective_boxing_actor_context.h"

namespace oneflow {

using namespace boxing::collective;

namespace {

OfCollectiveBoxingActorContext* GetOfCollectiveBoxingActorContext(KernelContext* kernel_ctx) {
  auto* actor_context_provider = CHECK_NOTNULL(dynamic_cast<ActorContextProvider*>(kernel_ctx));
  return CHECK_NOTNULL(
      dynamic_cast<OfCollectiveBoxingActorContext*>(actor_context_provider->GetActorContext()));
}

// class OfCollectiveBoxingKernelState final : public KernelState {
//  public:
//   OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingKernelState);
//   explicit OfCollectiveBoxingKernelState(const RankDesc& rank_desc)
//       : request_handle_(Singleton<Scheduler>::Get()->CreateRequestHandle(rank_desc)) {}
//   ~OfCollectiveBoxingKernelState() override {
//     Singleton<Scheduler>::Get()->DestroyRequestHandle(request_handle_);
//   }
//   RequestHandle* request_handle() { return request_handle_; }

//  private:
//   RequestHandle* request_handle_ = nullptr;
// };

class OfCollectiveBoxingGenericKernel final : public Kernel {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingGenericKernel);
  OfCollectiveBoxingGenericKernel() = default;
  ~OfCollectiveBoxingGenericKernel() override = default;

 private:
//   void VirtualKernelInit(KernelContext* ctx) override;
//   bool IsKernelLaunchSynchronized() const override { return false; }
  void ForwardDataContent(KernelContext* ctx) const override;
};

// void OfCollectiveBoxingGenericKernel::VirtualKernelInit(KernelContext* ctx) {
//   const RankDesc& rank_desc = this->op_conf().collective_boxing_generic_conf().rank_desc();
//   ctx->set_state(std::make_shared<OfCollectiveBoxingKernelState>(rank_desc));
// }

void OfCollectiveBoxingGenericKernel::ForwardDataContent(KernelContext* ctx) const {
  VLOG(1) << "Enter OfCollectiveBoxingGenericKernel::ForwardDataContent";
  Blob* in = ctx->BnInOp2Blob("in");
  Blob* out = ctx->BnInOp2Blob("out");
  AutoMemcpy(ctx->stream(), out, in);
  
  VLOG(1) << "OfCollectiveBoxingGenericKernel::ForwardDataContent Done";
}

REGISTER_KERNEL(OperatorConf::kOfCollectiveBoxingGenericConf, OfCollectiveBoxingGenericKernel);

}  // namespace

}  // namespace oneflow
