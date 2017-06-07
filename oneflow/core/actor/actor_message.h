#ifndef ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/register_warpper.h"

namespace oneflow {

enum class ActorCmd {
  kInitModel = 0
};

enum class ActorMsgType {
  kRegstMsg = 0,
  kCmdMsg = 1
};

class ActorMsg final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ActorMsg);
  ActorMsg();
  ~ActorMsg() = default;

  static ActorMsg BuildMsgForRegstReader(uint64_t reader_actor_id, Regst*);
  static ActorMsg BuildMsgForRegstWriter(uint64_t writer_actor_id, Regst*);

  // Getters
  uint64_t dst_actor_id() const { return dst_actor_id_; }
  ActorMsgType msg_type() const { return msg_type_; }
  std::shared_ptr<RegstWarpper> regst_warpper() const {
    CHECK(msg_type_ == ActorMsgType::kRegstMsg);
    return regst_warpper_;
  }
  ActorCmd actor_cmd() const {
    CHECK(msg_type_ == ActorMsgType::kCmdMsg);
    return actor_cmd_;
  }
  
 private:

  uint64_t dst_actor_id_;
  ActorMsgType msg_type_;

  std::shared_ptr<RegstWarpper> regst_warpper_;
  ActorCmd actor_cmd_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
