#ifndef ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_
#define ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_

#include "oneflow/core/common/util.h"

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

}  // namespace actor

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_NEW_ACTOR_H_
