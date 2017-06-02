#ifndef ONEFLOW_ACTOR_ACTOR_MESSAGE_H_
#define ONEFLOW_ACTOR_ACTOR_MESSAGE_H_

#include "oneflow/common/util.h"
#include "oneflow/register/register.h"

namespace oneflow {

class ActorMsg final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ActorMsg);
  ActorMsg();
  ~ActorMsg() = default;

  // Getters and Setters
  uint64_t dst_actor_id() const { return dst_actor_id_; }
  void set_dst_actor_id(uint64_t val) { dst_actor_id_ = val; }
  uint64_t piece_id() const { return piece_id_; }
  void set_piece_id(uint64_t val) { piece_id_ = val; }
  Regst* regst() const { return regst_; }
  void set_regst(Regst* val) { regst_ = val; }
  void* regst_dptr() const { return regst_dptr_; }
  void set_regst_dptr(void* val) { regst_dptr_ = val; }
  
 private:
  uint64_t dst_actor_id_;
  uint64_t piece_id_;
  Regst* regst_;
  void* regst_dptr_;

};

} // namespace oneflow

#endif // ONEFLOW_ACTOR_ACTOR_MESSAGE_H_
