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
#include "oneflow/core/lazy/actor/collective_boxing_actor_context.h"
#include "oneflow/core/job/collective_boxing/scheduler.h"

namespace oneflow {

using namespace boxing::collective;

void CollectiveBoxingActorContext::Init(const TaskProto& task_proto, StreamContext* stream_ctx) {
  stream_ctx_ = stream_ctx;
  task_proto_ = task_proto;
  scheduled_count_ = 0;
  completed_count_ = 0;
}

void CollectiveBoxingActorContext::AddCallback(std::function<void()> callback) {
  std::lock_guard<std::mutex> lock(mutex_);
  if (scheduled_count_ == completed_count_) {
    callback();
  } else {
    callbacks_.emplace_back(std::make_pair(scheduled_count_ - 1, std::move(callback)));
  }
}

void CollectiveBoxingActorContext::Schedule(RequestHandle* handle, const void* send_buff,
                                            void* recv_buff) {
  std::lock_guard<std::mutex> lock(mutex_);
  auto request = std::make_shared<boxing::collective::RuntimeRequestInfo>();
  request->send_buff = send_buff;
  request->recv_buff = recv_buff;
  const size_t schedule_id = scheduled_count_;
  request->callback = [schedule_id, this](const Maybe<void>& status) {
    CHECK(status.IsOk());
    this->SetCompleted(schedule_id);
  };
  Singleton<Scheduler>::Get()->Schedule(handle, request);
  scheduled_count_ += 1;
}

void CollectiveBoxingActorContext::SetCompleted(size_t schedule_id) {
  std::lock_guard<std::mutex> lock(mutex_);
  CHECK_EQ(schedule_id, completed_count_);
  while (!callbacks_.empty() && callbacks_.front().first == schedule_id) {
    callbacks_.front().second();
    callbacks_.pop_front();
  }
  completed_count_ += 1;
}

StreamContext* CollectiveBoxingActorContext::stream_ctx() const { return stream_ctx_; }

const TaskProto& CollectiveBoxingActorContext::task_proto() const { return task_proto_; }

REGISTER_ACTOR_CONTEXT(TaskType::kCollectiveBoxingGeneric, CollectiveBoxingActorContext);

}  // namespace oneflow
