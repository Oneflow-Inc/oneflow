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
#include "oneflow/core/actor/naive_actor.h"

namespace oneflow {

void NaiveActor::Act() { AsyncLaunchKernel(GenDefaultKernelCtx()); }

void NaiveActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  int64_t piece_id = GetPieceId4NaiveCurReadableDataRegst();
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    return true;
  });
}

REGISTER_ACTOR(TaskType::kSliceBoxing, NaiveActor);
REGISTER_ACTOR(TaskType::kBoxingIdentity, NaiveActor);
REGISTER_ACTOR(TaskType::kCollectiveBoxingPack, NaiveActor);
REGISTER_ACTOR(TaskType::kCollectiveBoxingUnpack, NaiveActor);
REGISTER_ACTOR(TaskType::kDecodeH2D, NaiveActor);

}  // namespace oneflow
