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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/job/runtime_job_descs.h"

namespace oneflow {

std::unique_ptr<ActorBase> NewActor(ActorContext* actor_ctx) {
  ActorBase* rptr = NewObj<int32_t, ActorBase>(actor_ctx->task_proto().task_type());
  const auto& job_descs = *Singleton<RuntimeJobDescs>::Get();
  rptr->Init(&job_descs.job_desc(actor_ctx->task_proto().job_id()), actor_ctx);
  return std::unique_ptr<ActorBase>(rptr);
}

}  // namespace oneflow
