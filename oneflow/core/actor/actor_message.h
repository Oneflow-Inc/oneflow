#ifndef ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/register_warpper.h"

namespace oneflow {

enum class ActorCmd {
  kInitializeModel = 0,
  kSendInitialModel,
  kStop
};

enum class ActorMsgType {
  kRegstMsg = 0,
  kCmdMsg
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
  // Setters
  void set_dst_actor_id(uint64_t val) {
    dst_actor_id_ = val;
  }
  void set_regst_warpper(std::shared_ptr<RegstWarpper> val) {
    msg_type_ = ActorMsgType::kRegstMsg;
    regst_warpper_ = val;
  }
  void set_actor_cmd(ActorCmd val) {
    msg_type_ = ActorMsgType::kCmdMsg;
    actor_cmd_ = val;
  }
  
 private:

  uint64_t dst_actor_id_;
  ActorMsgType msg_type_;

  std::shared_ptr<RegstWarpper> regst_warpper_;
  ActorCmd actor_cmd_;

};

} // namespace oneflow

#endif // ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
