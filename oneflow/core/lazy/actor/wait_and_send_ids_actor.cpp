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
#include "oneflow/core/lazy/actor/actor.h"
#include "oneflow/core/kernel/wait_and_send_ids_kernel.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

class WaitAndSendIdsActor final : public Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(WaitAndSendIdsActor);
  WaitAndSendIdsActor() : wait_and_send_ids_status_(nullptr) {}
  ~WaitAndSendIdsActor() = default;

 private:
  void VirtualActorInit(const TaskProto&) override;
  void Act() override;
  std::pair<RegstNameType, HashSet<std::string>> GetNaiveOrCustomizedConsumedRegstDescName()
      override {
    return std::make_pair(RegstNameType::kNaive, HashSet<std::string>{});
  }
  void VirtualAsyncSendNaiveProducedRegstMsgToConsumer() override;
  bool IsCustomizedReadReady() const override;
  bool IsCustomizedReadAlwaysUnReadyFromNow() const override { return !IsCustomizedReadReady(); }

  int HandlerWaitToStart(const ActorMsg&);

  WaitAndSendIdsStatus* wait_and_send_ids_status_;
};

void WaitAndSendIdsActor::VirtualActorInit(const TaskProto& task_proto) {
  CHECK_EQ(exec_kernel_vec().size(), 1);
  wait_and_send_ids_status_ = CHECK_NOTNULL(
      dynamic_cast<WaitAndSendIdsStatus*>(exec_kernel_vec().at(0).kernel_ctx->state().get()));
  wait_and_send_ids_status_->buffer_status_ = kBufferStatusSuccess;
  wait_and_send_ids_status_->in_id_ = 0;
  wait_and_send_ids_status_->out_idx_ = 0;
  wait_and_send_ids_status_->out_num_ = 0;
  OF_SET_MSG_HANDLER(&WaitAndSendIdsActor::HandlerWaitToStart);
}

void WaitAndSendIdsActor::Act() {
  CHECK_LE(wait_and_send_ids_status_->out_idx_, wait_and_send_ids_status_->out_num_);
  AsyncLaunchKernel();
}

void WaitAndSendIdsActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (wait_and_send_ids_status_->buffer_status_ == kBufferStatusSuccess) {
    HandleProducedNaiveDataRegstToConsumer();
  }
}

bool WaitAndSendIdsActor::IsCustomizedReadReady() const {
  return wait_and_send_ids_status_->buffer_status_ == kBufferStatusSuccess;
}

int WaitAndSendIdsActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&WaitAndSendIdsActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kWaitAndSendIds, WaitAndSendIdsActor);

}  // namespace oneflow
