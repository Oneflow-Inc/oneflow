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
#ifndef ONEFLOW_CORE_LAZY_ACTOR_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_
#define ONEFLOW_CORE_LAZY_ACTOR_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_

#include "oneflow/core/lazy/actor/actor_context.h"
#include "oneflow/core/job/collective_boxing/scheduler.h"

namespace oneflow {

class CollectiveBoxingActorContext : public ActorContext {
 public:
  OF_DISALLOW_COPY_AND_MOVE(CollectiveBoxingActorContext);
  CollectiveBoxingActorContext() = default;
  ~CollectiveBoxingActorContext() override = default;

  void Init(const TaskProto& task_proto, StreamContext* stream_ctx) override;
  void AddCallback(std::function<void()> callback) override;
  void Schedule(boxing::collective::RequestHandle* handle, const void* send_buff, void* recv_buff);
  void SetCompleted(size_t schedule_id);

  StreamContext* stream_ctx() const override;
  const TaskProto& task_proto() const override;

 private:
  StreamContext* stream_ctx_{};
  TaskProto task_proto_{};
  size_t scheduled_count_{};
  size_t completed_count_{};
  std::mutex mutex_;
  std::deque<std::pair<size_t, std::function<void()>>> callbacks_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_LAZY_ACTOR_COLLECTIVE_BOXING_ACTOR_CONTEXT_H_
