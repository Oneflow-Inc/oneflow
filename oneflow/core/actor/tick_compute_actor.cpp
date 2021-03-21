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
#include "oneflow/core/actor/tick_compute_actor.h"

namespace oneflow {

void TickComputeActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  OF_SET_MSG_HANDLER(&TickComputeActor::HandlerNormal);
}

void TickComputeActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([&](Regst* out_regst) {
    out_regst->set_piece_id(piece_id_++);
    return true;
  });
}

REGISTER_ACTOR(kTick, TickComputeActor);
REGISTER_ACTOR(kDeviceTick, TickComputeActor);
REGISTER_ACTOR(kSrcSubsetTick, TickComputeActor);
REGISTER_ACTOR(kDstSubsetTick, TickComputeActor);

}  // namespace oneflow
