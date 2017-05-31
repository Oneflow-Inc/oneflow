#ifndef ONEFLOW_ACTOR_ACTOR_MESSAGE_H_
#define ONEFLOW_ACTOR_ACTOR_MESSAGE_H_

#include "common/util.h"
#include "register/register.h"

namespace oneflow {

class ActorMsg final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ActorMsg);
  ActorMsg();
  ~ActorMsg() = default;

  // Getters
  uint64_t dst_actor_id() const { return dst_actor_id_; }
  Regst* regst() const { return regst_; }

  // Setters
  void set_dst_actor_id(uint64_t val) { dst_actor_id_ = val; }
  void set_regst(Regst* val) { regst_ = val; }

 private:
  uint64_t dst_actor_id_;
  Regst* regst_;

};

} // namespace oneflow

#endif // ONEFLOW_ACTOR_ACTOR_MESSAGE_H_
