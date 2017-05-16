#ifndef ONEFLOW_ACTOR_ACTOR_H_
#define ONEFLOW_ACTOR_ACTOR_H_

#include "common/util.h"
#include "actor/task.pb.h"

namespace oneflow {

class Actor {
 public:
  OF_DISALLOW_COPY_AND_MOVE(Actor);
  virtual ~Actor() = default;

  virtual void Init(const TaskProto&);
  virtual void ProcessMsg(const ActorMsg&);

 protected:
  Actor() = default;

 private:

};

} // namespace oneflow

#endif // ONEFLOW_ACTOR_ACTOR_H_
