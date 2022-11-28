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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_ACTOR_CONTEXT_H_
#define ONEFLOW_CORE_LAZY_ACTOR_ACTOR_CONTEXT_H_

#include "oneflow/core/lazy/stream_context/include/stream_context.h"
#include "oneflow/core/job/task.pb.h"

namespace oneflow {

class ActorContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorContext);
  ActorContext() = default;
  virtual ~ActorContext() = default;

  virtual void Init(const TaskProto& task_proto, StreamContext* stream_ctx) = 0;
  virtual void AddCallback(std::function<void()> callback) = 0;

  virtual StreamContext* stream_ctx() const = 0;
  virtual const TaskProto& task_proto() const = 0;
};

class ActorContextProvider {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorContextProvider);
  ActorContextProvider() = default;
  virtual ~ActorContextProvider() = default;

  virtual ActorContext* GetActorContext() const = 0;
};

std::unique_ptr<ActorContext> NewActorContext(const TaskProto& task_proto,
                                              StreamContext* stream_ctx);

#define REGISTER_ACTOR_CONTEXT(task_type, ActorContextType) \
  REGISTER_CLASS(int32_t, task_type, ActorContext, ActorContextType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_ACTOR_CONTEXT_H_
