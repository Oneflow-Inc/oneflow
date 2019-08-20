#ifndef ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/register_manager.h"
#include "oneflow/core/kernel/kernel.h"
#include "oneflow/core/kernel/kernel_context.h"
#include "oneflow/core/thread/thread_context.h"
#include "oneflow/core/job/task.pb.h"

namespace oneflow {

namespace actor {

using MsgHandler = std::function<int(const ActorMsg&)>;

class NewActor {
 public:
  NewActor() = default;
  ~NewActor() = default;
  OF_DISALLOW_COPY_AND_MOVE(NewActor);

  virtual void Init(const TaskProto&, const ThreadCtx&) = 0;

  // 1: success, and actor finish
  // 0: success, and actor not finish
  int ProcessMsg(const ActorMsg& msg) { return (this->msg_handler_)(msg); }

 protected:
  void set_msg_handler(MsgHandler val) { msg_handler_ = val; }

 private:
  MsgHandler msg_handler_;
};

std::unique_ptr<NewActor> CreateNewActor(const TaskProto&, const ThreadCtx&);

}  // namespace actor

#define REGISTER_NEW_ACTOR(task_type, ActorType) REGISTER_CLASS(task_type, NewActor, ActorType)

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_
