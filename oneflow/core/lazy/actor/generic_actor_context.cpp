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
#include "oneflow/core/lazy/actor/generic_actor_context.h"

namespace oneflow {

void GenericActorContext::Init(const TaskProto& task_proto, StreamContext* stream_ctx) {
  stream_ctx_ = stream_ctx;
  task_proto_ = task_proto;
}

void GenericActorContext::AddCallback(std::function<void()> callback) {
  CHECK_JUST(stream_ctx_->AddCallback(std::move(callback)));
}

StreamContext* GenericActorContext::stream_ctx() const { return stream_ctx_; }

const TaskProto& GenericActorContext::task_proto() const { return task_proto_; }

}  // namespace oneflow
