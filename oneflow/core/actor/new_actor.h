#ifndef ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_

#include "oneflow/core/actor/op_actor_context.h"

namespace oneflow {

namespace actor {

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

class OpActor final : public NewActor {
 public:
  static int HandlerNormal(OpActor*, const ActorMsg&);
  static int HandlerZombie(OpActor*, const ActorMsg&);

  OpActor() = default;
  ~OpActor() = default;
  OF_DISALLOW_COPY_AND_MOVE(OpActor);

  void Init(const TaskProto& task, const ThreadCtx& thread_ctx) override {
    op_actor_ctx_.reset(CreateOpActorCtx(task.task_type()));
    op_actor_ctx_->Init(task, thread_ctx);
    set_msg_handler(op_actor_ctx_->initial_msg_handler());
  }

  OpActorCtx* op_actor_ctx() { return op_actor_ctx_.get(); }

 private:
  std::unique_ptr<OpActorCtx> op_actor_ctx_;
};

}  // namespace actor

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_
