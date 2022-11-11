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
#include "oneflow/core/thread/thread.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/lazy/actor/actor.h"
#include "oneflow/core/lazy/actor/light_actor.h"
#include "oneflow/core/profiler/profiler.h"
#include "oneflow/core/lazy/stream_context/include/stream_context.h"
#include "oneflow/core/framework/to_string.h"
#include "oneflow/core/lazy/stream_context/include/generic_stream_context.h"
#include "oneflow/core/job/lazy_mode.h"

namespace oneflow {

Thread::Thread(const StreamId& stream_id) : thrd_id_(EncodeStreamIdToInt64(stream_id)) {
  local_msg_queue_enabled_ = ParseBooleanFromEnv("ONEFLOW_THREAD_ENABLE_LOCAL_MESSAGE_QUEUE", true);
  light_actor_enabled_ = ParseBooleanFromEnv("ONEFLOW_ACTOR_ENABLE_LIGHT_ACTOR", true);
  if (IsClassRegistered<int, StreamContext, const StreamId&>(stream_id.device_id().device_type(),
                                                             stream_id)) {
    stream_ctx_.reset(NewObj<int, StreamContext, const StreamId&>(
        stream_id.device_id().device_type(), stream_id));
  } else {
    stream_ctx_.reset(new GenericStreamContext(stream_id));
  }

  actor_thread_ = std::thread([this, stream_id]() {
    LazyMode::Guard guard(true);
    OF_PROFILER_NAME_THIS_HOST_THREAD("_" + ToString(stream_id.device_id().device_type())
                                      + std::to_string(stream_id.device_id().device_index())
                                      + "_actor");
    CHECK_JUST(stream_ctx_->stream()->OnExecutionContextSetup());
    PollMsgChannel();
    CHECK_JUST(stream_ctx_->stream()->OnExecutionContextTeardown());
  });
}

Thread::~Thread() {
  actor_thread_.join();
  CHECK(id2task_.empty());
  msg_channel_.Close();
}

void Thread::AddTask(const TaskProto& task) {
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  CHECK(id2task_.emplace(task.task_id(), task).second);
}

void Thread::PollMsgChannel() {
  while (true) {
    if (local_msg_queue_.empty()) {
      CHECK_EQ(msg_channel_.ReceiveMany(&local_msg_queue_), kChannelStatusSuccess);
    }
    ActorMsg msg = std::move(local_msg_queue_.front());
    local_msg_queue_.pop();
    if (msg.msg_type() == ActorMsgType::kCmdMsg) {
      if (msg.actor_cmd() == ActorCmd::kStopThread) {
        CHECK(id2actor_ptr_.empty())
            << " RuntimeError! Thread: " << thrd_id_
            << " NOT empty when stop with actor num: " << id2actor_ptr_.size();
        break;
      } else if (msg.actor_cmd() == ActorCmd::kConstructActor) {
        ConstructActor(msg.dst_actor_id());
        continue;
      } else {
        // do nothing
      }
    }
    int64_t actor_id = msg.dst_actor_id();
    auto actor_it = id2actor_ptr_.find(actor_id);
    CHECK(actor_it != id2actor_ptr_.end());
    int process_msg_ret = actor_it->second.second->ProcessMsg(msg);
    if (process_msg_ret == 1) {
      VLOG(3) << "thread " << thrd_id_ << " deconstruct actor " << actor_id;
      auto job_id_it = id2job_id_.find(actor_id);
      const int64_t job_id = job_id_it->second;
      id2job_id_.erase(job_id_it);
      id2actor_ptr_.erase(actor_it);
      Singleton<RuntimeCtx>::Get()->DecreaseCounter(GetRunningActorCountKeyByJobId(job_id));
    } else {
      CHECK_EQ(process_msg_ret, 0);
    }
  }
}

void Thread::ConstructActor(int64_t actor_id) {
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  auto task_it = id2task_.find(actor_id);
  const TaskProto& task = task_it->second;
  std::unique_ptr<ActorContext> actor_ctx = NewActorContext(task, stream_ctx_.get());
  CHECK(actor_ctx);
  std::unique_ptr<ActorBase> actor_ptr;
  if (light_actor_enabled_) { actor_ptr = TryNewLightActor(actor_ctx.get()); }
  if (!actor_ptr) {
    actor_ptr = NewActor(actor_ctx.get());
    VLOG(3) << "Thread " << thrd_id_ << " construct Actor " << TaskType_Name(task.task_type())
            << " " << actor_id;
  } else {
    VLOG(3) << "Thread " << thrd_id_ << " construct LightActor " << TaskType_Name(task.task_type())
            << " " << actor_id;
  }
  CHECK(id2actor_ptr_.emplace(actor_id, std::make_pair(std::move(actor_ctx), std::move(actor_ptr)))
            .second);
  CHECK(id2job_id_.emplace(actor_id, task.job_id()).second);
  id2task_.erase(task_it);
  Singleton<RuntimeCtx>::Get()->DecreaseCounter("constructing_actor_cnt");
}

}  // namespace oneflow
