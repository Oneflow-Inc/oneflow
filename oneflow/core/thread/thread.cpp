#include "oneflow/core/thread/thread.h"

namespace std {

template<>
struct hash<::oneflow::TaskType> {
  std::size_t operator()(const ::oneflow::TaskType& task_type) const {
    return std::hash<int>()(static_cast<size_t>(task_type));
  }
};

}  // namespace std

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

void Thread::PollMsgChannel(const ThreadCtx& thread_ctx) {
  ActorMsg msg;
  while (true) {
    CHECK_EQ(msg_channel_.Receive(&msg), kChannelStatusSuccess);
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
    int process_msg_ret = -1;
    if (actor_it == id2actor_ptr_.end()) {
      process_msg_ret = id2new_actor_ptr_.at(actor_id)->ProcessMsg(msg);
    } else {
      process_msg_ret = actor_it->second->ProcessMsg(msg);
    }
    if (process_msg_ret == 1) {
      LOG(INFO) << "thread " << thrd_id_ << " deconstruct actor " << actor_id;
      if (actor_it != id2actor_ptr_.end()) {
        id2actor_ptr_.erase(actor_it);
      } else {
        id2new_actor_ptr_.erase(actor_id);
      }
      Global<RuntimeCtx>::Get()->DecreaseCounter("running_actor_cnt");
    } else {
      CHECK_EQ(process_msg_ret, 0);
    }
  }
}

namespace {
const HashSet<TaskType>& TaskWithNewActor() {
  static HashSet<TaskType> tasks = {};
  return tasks;
}
}  // namespace

void Thread::ConstructActor(int64_t actor_id, const ThreadCtx& thread_ctx) {
  LOG(INFO) << "thread " << thrd_id_ << " construct actor " << actor_id;
  std::unique_lock<std::mutex> lck(id2task_mtx_);
  auto task_it = id2task_.find(actor_id);
  if (IsKeyFound(TaskWithNewActor(), task_it->second.task_type())) {
    CHECK(id2new_actor_ptr_.emplace(actor_id, actor::ConstructNewActor(task_it->second, thread_ctx))
              .second);
  } else {
    CHECK(id2actor_ptr_.emplace(actor_id, NewActor(task_it->second, thread_ctx)).second);
  }
  id2task_.erase(task_it);
  Global<RuntimeCtx>::Get()->DecreaseCounter("constructing_actor_cnt");
}

}  // namespace oneflow
