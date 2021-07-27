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
#include "oneflow/core/actor/wait_and_send_ids_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void WaitAndSendIdsCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  wait_and_send_ids_status_.buffer_status_ = kBufferStatusSuccess;
  wait_and_send_ids_status_.in_id_ = 0;
  wait_and_send_ids_status_.out_idx_ = 0;
  wait_and_send_ids_status_.out_num_ = 0;
  cur_piece_id_ = -1;
  OF_SET_MSG_HANDLER(&WaitAndSendIdsCompActor::HandlerWaitToStart);
}

void WaitAndSendIdsCompActor::Act() {
  CHECK_LE(wait_and_send_ids_status_.out_idx_, wait_and_send_ids_status_.out_num_);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &wait_and_send_ids_status_;
  AsyncLaunchKernel(kernel_ctx);
}

void WaitAndSendIdsCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (wait_and_send_ids_status_.buffer_status_ == kBufferStatusSuccess) {
    HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
      regst->set_piece_id(++cur_piece_id_);
      return true;
    });
  }
}

bool WaitAndSendIdsCompActor::IsCustomizedReadReady() const {
  return wait_and_send_ids_status_.buffer_status_ == kBufferStatusSuccess;
}

int WaitAndSendIdsCompActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&WaitAndSendIdsCompActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kWaitAndSendIds, WaitAndSendIdsCompActor);

}  // namespace oneflow
