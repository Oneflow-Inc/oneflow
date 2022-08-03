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
#include "oneflow/core/lazy/actor/naive_actor.h"

namespace oneflow {

void NaiveActor::Act() {
  AsyncLaunchKernel([&](int64_t regst_desc_id) -> Regst* { return nullptr; });
}

void NaiveActor::VirtualActorInit(const TaskProto&) {
  OF_SET_MSG_HANDLER(&NaiveActor::HandlerNormal);
}

REGISTER_ACTOR(TaskType::kNormalForward, NaiveActor);
REGISTER_ACTOR(TaskType::kDistributeConcat, NaiveActor);
REGISTER_ACTOR(TaskType::kDistributeSplit, NaiveActor);
REGISTER_ACTOR(TaskType::kSliceBoxing, NaiveActor);
REGISTER_ACTOR(TaskType::kBoxingIdentity, NaiveActor);
REGISTER_ACTOR(TaskType::kCollectiveBoxingPack, NaiveActor);
REGISTER_ACTOR(TaskType::kCollectiveBoxingUnpack, NaiveActor);
REGISTER_ACTOR(TaskType::kNcclSendRecvBoxing, NaiveActor);
REGISTER_ACTOR(TaskType::kDecodeH2D, NaiveActor);
REGISTER_ACTOR(TaskType::kCriticalSectionWaitTick, NaiveActor);
REGISTER_ACTOR(TaskType::kCopyHd, NaiveActor);
REGISTER_ACTOR(TaskType::kCollectiveBoxingGeneric, NaiveActor);
}  // namespace oneflow
