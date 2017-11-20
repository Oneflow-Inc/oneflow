#include "oneflow/core/thread/thread.h"

namespace oneflow {

Thread::~Thread() {
  actor_thread_.join();
  CHECK(id2task_.empty());
  msg_channel_.CloseSendEnd();
  msg_channel_.CloseReceiveEnd();
}

void Thread::AddTask(const TaskProto& task) {
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  CHECK(id2task_.emplace(task.task_id(), task).second);
}

void Thread::PollMsgChannel(const ThreadCtx& thread_ctx) {
  ActorMsg msg;
  while (true) {
    CHECK_EQ(msg_channel_.Receive(&msg), 0);
    if (msg.msg_type() == ActorMsgType::kCmdMsg) {
      if (msg.actor_cmd() == ActorCmd::kStopThread) {
        CHECK(id2actor_ptr_.empty());
        break;
      } else if (msg.actor_cmd() == ActorCmd::kActivateActor) {
        ActivateActor(msg.dst_actor_id(), thread_ctx);
        continue;
      } else {
        // do nothing
      }
    }
    int64_t actor_id = msg.dst_actor_id();
    auto actor_it = id2actor_ptr_.find(actor_id);
    int process_msg_ret = actor_it->second->ProcessMsg(msg);
    if (process_msg_ret == 1) {
      LOG(INFO) << "thread " << thrd_id_ << " deconstruct actor " << actor_id;
      id2actor_ptr_.erase(actor_it);
      RuntimeCtx::Singleton()->mut_active_actor_cnt().MinusOne();
    } else {
      CHECK_EQ(process_msg_ret, 0);
    }
  }
}

void Thread::ActivateActor(int64_t actor_id, const ThreadCtx& thread_ctx) {
  LOG(INFO) << "thread " << thrd_id_ << " construct actor " << actor_id;
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  auto task_it = id2task_.find(actor_id);
  CHECK(id2actor_ptr_
            .emplace(actor_id, ConstructActor(task_it->second, thread_ctx))
            .second);
  id2task_.erase(task_it);
  RuntimeCtx::Singleton()->mut_inactive_actor_cnt().MinusOne();
}

}  // namespace oneflow
