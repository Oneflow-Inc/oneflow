#ifndef ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

enum class ActorCmd {
  kInitializeModel = 0,  // MdUpdt Actor
  kSendInitialModel,     // MdUpdt Actor
  kEORD,                 // End Of Register Desc, All Actor except Source Actor
  kStart,                // Source Actor
  kStopThread,
  kActivateActor
};

OF_DECLARE_ENUM_TO_OSTREAM_FUNC(ActorCmd);

enum class ActorMsgType { kRegstMsg = 0, kCmdMsg };

OF_DECLARE_ENUM_TO_OSTREAM_FUNC(ActorMsgType);

class ActorMsg final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ActorMsg);
  ActorMsg();
  ~ActorMsg() = default;

  static ActorMsg BuildReadableRegstMsg(int64_t writer_actor_id,
                                        int64_t reader_actor_id, Regst*);
  static ActorMsg BuildRegstMsgToProducer(int64_t writer_actor_id,
                                          int64_t reader_actor_id, Regst*);
  static ActorMsg BuildRegstMsgToProducer(int64_t writer_actor_id,
                                          int64_t reader_actor_id, Regst*,
                                          int64_t piece_id);

  // Getters
  int64_t src_actor_id() const { return src_actor_id_; }
  int64_t dst_actor_id() const { return dst_actor_id_; }
  ActorMsgType msg_type() const { return msg_type_; }
  Regst* regst() const {
    CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
    return regst_;
  }
  ActorCmd actor_cmd() const {
    CHECK_EQ(msg_type_, ActorMsgType::kCmdMsg);
    return actor_cmd_;
  }

  // Setters
  void set_src_actor_id(int64_t val) { src_actor_id_ = val; }
  void set_dst_actor_id(int64_t val) { dst_actor_id_ = val; }
  void set_regst(Regst* val) {
    msg_type_ = ActorMsgType::kRegstMsg;
    regst_ = val;
  }
  void set_actor_cmd(ActorCmd val) {
    msg_type_ = ActorMsgType::kCmdMsg;
    actor_cmd_ = val;
  }

  void set_piece_id(int64_t piece_id) { piece_id_ = piece_id; }
  int64_t piece_id() const { return piece_id_; }

  // Serialize
  template<typename StreamT>
  void Serialize(StreamT& out_stream) const {
    TODO();
  }
  template<typename StreamT>
  void Deserialize(StreamT& in_stream) {
    TODO();
  }

 private:
  int64_t src_actor_id_;
  int64_t dst_actor_id_;
  ActorMsgType msg_type_;

  Regst* regst_;
  ActorCmd actor_cmd_;

  int64_t piece_id_;
};

template<typename StreamT>
StreamT& operator<<(StreamT& out_stream, const ActorMsg& msg) {
  msg.Serialize(out_stream);
}

template<typename StreamT>
StreamT& operator>>(StreamT& in_stream, const ActorMsg& msg) {
  msg.Deserialize(in_stream);
}

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
