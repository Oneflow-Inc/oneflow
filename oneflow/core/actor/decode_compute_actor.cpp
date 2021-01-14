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
#include "oneflow/core/actor/decode_compute_actor.h"

namespace oneflow {

void DecodeCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  piece_id_ = 0;
  OF_SET_MSG_HANDLER(&DecodeCompActor::HandlerNormal);
}

void DecodeCompActor::Act() {
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  AsyncLaunchKernel(kernel_ctx);
}

void DecodeCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  Regst* in_regst = GetNaiveCurReadable("record");
  HandleProducedNaiveDataRegstToConsumer([this, in_regst](Regst* regst) {
    regst->set_piece_id(in_regst->piece_id());
    return true;
  });
}

void DecodeCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  HandleConsumedNaiveDataRegstToProducer([this](Regst*) { return true; });
}

REGISTER_ACTOR(kDecode, DecodeCompActor);

}  // namespace oneflow
