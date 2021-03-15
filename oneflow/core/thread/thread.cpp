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
#include "oneflow/core/control/global_process_ctx.h"
#include "oneflow/core/common/id_util.h"
#include "oneflow/core/graph/id_serialization.h"
#include "oneflow/core/thread/thread.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/job/global_for.h"
#include "oneflow/core/actor/actor.h"
#include "oneflow/core/job/global_for.h"

namespace oneflow {

Thread::~Thread() {
  actor_thread_.join();
  CHECK(id2task_.empty());
  msg_channel_.Close();
}

void Thread::AddTask(const TaskProto& task) {
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  CHECK(id2task_.emplace(task.task_id(), task).second);
}

void Thread::EnqueueActorMsg(const ActorMsg& msg) {
  if (Global<ResourceDesc, ForSession>::Get()->thread_enable_local_message_queue()
      && std::this_thread::get_id() == actor_thread_.get_id()) {
    local_msg_queue_.push(msg);
  } else {
    msg_channel_.Send(msg);
  }
}

void Thread::PollMsgChannel(const ThreadCtx& thread_ctx) {
  while (true) {
    if (local_msg_queue_.empty()) {
      CHECK_EQ(msg_channel_.ReceiveMany(&local_msg_queue_), kChannelStatusSuccess);
    }
    ActorMsg msg = std::move(local_msg_queue_.front());
    local_msg_queue_.pop();
    if (msg.msg_type() == ActorMsgType::kCmdMsg) {
      if (msg.actor_cmd() == ActorCmd::kStopThread) {
        CHECK(id2actor_ptr_.empty());
        break;
      } else if (msg.actor_cmd() == ActorCmd::kConstructActor) {
        ConstructActor(msg.dst_actor_id(), thread_ctx);
        continue;
      } else {
        // do nothing
      }
    }
    int64_t actor_id = msg.dst_actor_id();
    auto actor_it = id2actor_ptr_.find(actor_id);
    CHECK(actor_it != id2actor_ptr_.end());
    int process_msg_ret = actor_it->second->ProcessMsg(msg);
    if (process_msg_ret == 1) {
      LOG(INFO) << "thread " << thrd_id_ << " deconstruct actor " << actor_id;
      id2actor_ptr_.erase(actor_it);
      Global<RuntimeCtx>::Get()->DecreaseCounter("running_actor_cnt");
    } else {
      CHECK_EQ(process_msg_ret, 0);
    }
  }
}

void Thread::ConstructActor(int64_t actor_id, const ThreadCtx& thread_ctx) {
  LOG(INFO) << "thread " << thrd_id_ << " construct actor " << actor_id;
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  auto task_it = id2task_.find(actor_id);
  CHECK(id2actor_ptr_.emplace(actor_id, NewActor(task_it->second, thread_ctx)).second);
  id2task_.erase(task_it);
  Global<RuntimeCtx>::Get()->DecreaseCounter("constructing_actor_cnt");
}

}  // namespace oneflow
