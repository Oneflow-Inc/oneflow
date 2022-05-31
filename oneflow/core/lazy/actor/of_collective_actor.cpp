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
#include "oneflow/core/lazy/actor/of_collective_actor.h"

namespace oneflow {

// void OfCollectiveActor::Act() {
//   AsyncLaunchKernel([&](int64_t regst_desc_id) -> Regst* { return nullptr; });
// }

void OfCollectiveActor::Init(const JobDesc* job_desc, ActorContext* actor_ctx) {
  actor_ctx_ = actor_ctx;
  const TaskProto& task_proto = actor_ctx->task_proto();
  actor_id_ = task_proto.task_id();
  thrd_id_ = ThrdId4ActorId(actor_id_);
  job_id_ = task_proto.job_id();
}

int OfCollectiveActor::HandlerNormal(const ActorMsg& msg) {

  return 0;
}

REGISTER_ACTOR(TaskType::kOfCollectiveBoxingGeneric, OfCollectiveActor);
}  // namespace oneflow
