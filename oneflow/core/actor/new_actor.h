#ifndef ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_

#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/actor/op_actor_context.h"

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
#define OF_SET_MSG_HANDLER(val)                                   \
  do {                                                            \
    LOG(INFO) << "actor " << actor_id() << " switch to " << #val; \
    set_msg_handler(static_cast<MsgHandler>(val));                \
  } while (0)

 private:
  MsgHandler msg_handler_;
};

struct HandlerUtil final {
  static int HandlerNormal(const ActorMsg&);
  static int HandlerEord(const ActorMsg&);
};

class OpActor final : public NewActor {
 public:
  OpActor() = default;
  ~OpActor() = default;
  OF_DISALLOW_COPY_AND_MOVE(OpActor);

  void Init(const TaskProto& task, const ThreadCtx& thread_ctx) override {
    op_actor_ctx_.reset(CreateOpActorCtx(task.task_type()));
    op_actor_ctx_->Init(task, thread_ctx);
    OF_SET_MSG_HANDLER(op_actor_ctx_->initial_msg_handler());
  }

 private:
  std::unique_ptr<OpActorCtx> op_actor_ctx_;
};

}

}

#endif // ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_
