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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_
#define ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_

#include "oneflow/core/lazy/actor/actor_context.h"
#include "oneflow/core/job/of_collective_boxing/collective_manager.h"

namespace oneflow {

class OfCollectiveBoxingActorContext : public ActorContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(OfCollectiveBoxingActorContext);
  OfCollectiveBoxingActorContext() : actor_id_(0) {};
  ~OfCollectiveBoxingActorContext() override = default;

  void Init(const TaskProto& task_proto, StreamContext* stream_ctx) override;
  void AddCallback(std::function<void()> callback) override;

  StreamContext* stream_ctx() const override;
  const TaskProto& task_proto() const override;

  void set_actor_id(int64_t actor_id) { actor_id_ = actor_id; }
  const int64_t actor_id() { return actor_id_; }

 private:
  StreamContext* stream_ctx_{};
  TaskProto task_proto_{};
  int64_t actor_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_OF_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_
