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
#include "oneflow/core/lazy/actor/actor_context.h"
#include "oneflow/core/lazy/actor/generic_actor_context.h"

namespace oneflow {

std::unique_ptr<ActorContext> NewActorContext(const TaskProto& task_proto,
                                              StreamContext* stream_ctx) {
  ActorContext* ctx = nullptr;
  if (IsClassRegistered<int32_t, ActorContext>(task_proto.task_type())) {
    ctx = NewObj<int32_t, ActorContext>(task_proto.task_type());
  } else {
    ctx = new GenericActorContext();
  }
  ctx->Init(task_proto, stream_ctx);
  return std::unique_ptr<ActorContext>(ctx);
}

}  // namespace oneflow
