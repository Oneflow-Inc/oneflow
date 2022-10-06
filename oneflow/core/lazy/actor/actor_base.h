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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_ACTOR_BASE_H_
#define ONEFLOW_CORE_LAZY_ACTOR_ACTOR_BASE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/lazy/actor/actor_context.h"

namespace oneflow {

class JobDesc;
class TaskProto;
class StreamContext;
class ActorMsg;

class ActorBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(ActorBase);
  ActorBase() = default;
  virtual ~ActorBase() = default;

  virtual void Init(const JobDesc* job_desc, ActorContext* actor_ctx) = 0;

  // 1: success, and actor finish
  // 0: success, and actor not finish
  virtual int ProcessMsg(const ActorMsg& msg) = 0;
};

std::unique_ptr<ActorBase> NewActor(ActorContext* actor_ctx);

#define REGISTER_ACTOR(task_type, ActorType) \
  REGISTER_CLASS(int32_t, task_type, ActorBase, ActorType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_ACTOR_BASE_H_
