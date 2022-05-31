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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_ACTOR_H_
#define ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_ACTOR_H_

#include "oneflow/core/lazy/actor/actor_base.h"
#include "oneflow/core/lazy/actor/actor_message_bus.h"
#include "oneflow/core/job/task.pb.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/lazy/actor/register_slot.h"
#include "oneflow/core/lazy/actor/of_collective_boxing_actor_context.h"

namespace oneflow {

class OfCollectiveActor final: public ActorBase {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveActor);
  OfCollectiveActor() = default;
  ~OfCollectiveActor() override = default;

  void Init(const JobDesc* job_desc, ActorContext* actor_ctx) override;

  // 1: success, and actor finish
  // 0: success, and actor not finish
  int ProcessMsg(const ActorMsg& msg) override { return (this->*msg_handler_)(msg); }

 private:
  using MsgHandler = int (OfCollectiveActor::*)(const ActorMsg&);
    // Msg Handler
  void set_msg_handler(MsgHandler val) { msg_handler_ = val; }
#define OF_SET_MSG_HANDLER(val)                                 \
  do {                                                          \
    VLOG(3) << "actor " << actor_id() << " switch to " << #val; \
    set_msg_handler(static_cast<MsgHandler>(val));              \
  } while (0)

  // Common Handlers and related virtual method
  int HandlerNormal(const ActorMsg& msg);

  
  // void Act();
  // void AsyncLaunchKernel(std::function<Regst*(int64_t)> Regst4RegstDescId);

  ActorContext* actor_ctx_;  
  int64_t actor_id_;
  int64_t thrd_id_;
  int64_t job_id_;
  MsgHandler msg_handler_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_ACTOR_H_
