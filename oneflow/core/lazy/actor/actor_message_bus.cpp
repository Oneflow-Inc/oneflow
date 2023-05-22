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
#include "oneflow/core/lazy/actor/actor_message_bus.h"
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

void ActorMsgBus::SendMsg(const ActorMsg& msg) {
  int64_t dst_machine_id = MachineId4ActorId(msg.dst_actor_id());
  if (dst_machine_id == GlobalProcessCtx::Rank()) {
    SendMsgWithoutCommNet(msg);
  } else {
    if (msg.IsDataRegstMsgToConsumer()) {
      int64_t comm_net_sequence;
      {
        std::unique_lock<std::mutex> lock(
            regst_desc_id_dst_actor_id2comm_net_sequence_number_mutex_);
        int64_t& comm_net_sequence_ref =
            regst_desc_id_dst_actor_id2comm_net_sequence_number_[std::make_pair(
                msg.regst_desc_id(), msg.dst_actor_id())];
        comm_net_sequence = comm_net_sequence_ref;
        comm_net_sequence_ref += 1;
      }
      ActorMsg new_msg = msg;
      new_msg.set_comm_net_sequence_number(comm_net_sequence);
      Singleton<CommNet>::Get()->SendActorMsg(dst_machine_id, new_msg);
    } else {
      Singleton<CommNet>::Get()->SendActorMsg(dst_machine_id, msg);
    }
  }
}

void ActorMsgBus::SendMsgWithoutCommNet(const ActorMsg& msg) {
  CHECK_EQ(MachineId4ActorId(msg.dst_actor_id()), GlobalProcessCtx::Rank());
  int64_t thrd_id = ThrdId4ActorId(msg.dst_actor_id());
  Singleton<ThreadMgr>::Get()->GetThrd(thrd_id)->EnqueueActorMsg(msg);
}

void ActorMsgBus::SendMsgsWithoutCommNet(const ActorMsg* msgs, size_t n, int64_t thrd_id) {
  Singleton<ThreadMgr>::Get()->GetThrd(thrd_id)->EnqueueActorMsg(msgs, msgs + n);
}

}  // namespace oneflow
