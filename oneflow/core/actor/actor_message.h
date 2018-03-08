#ifndef ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/register.h"

namespace oneflow {

enum class ActorCmd {
  kInitModel = 0,     // MdUpdt Actor
  kSendInitialModel,  // MdUpdt Actor
  kStart,             // Source Actor
  kStopThread,
  kConstructActor
};

OF_DECLARE_ENUM_TO_OSTREAM_FUNC(ActorCmd);

enum class ActorMsgType { kRegstMsg = 0, kEordMsg, kCmdMsg };

OF_DECLARE_ENUM_TO_OSTREAM_FUNC(ActorMsgType);

class ActorMsg final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ActorMsg);
  ActorMsg() = default;
  ~ActorMsg() = default;

  // Build Msg
  static ActorMsg BuildRegstMsgToConsumer(int64_t producer, int64_t consumer,
                                          Regst*);
  static ActorMsg BuildRegstMsgToProducer(int64_t consumer, int64_t producer,
                                          Regst*);
  static ActorMsg BuildEordMsg(int64_t consumer, int64_t regst_desc_id);
  static ActorMsg BuildCommandMsg(int64_t dst_actor_id, ActorCmd cmd);

  // Getters
  int64_t SrcMachineId() const;
  int64_t src_actor_id() const { return src_actor_id_; }
  int64_t dst_actor_id() const { return dst_actor_id_; }
  ActorMsgType msg_type() const { return msg_type_; }
  ActorCmd actor_cmd() const;
  Regst* regst() const;
  int64_t piece_id() const;
  int64_t act_id() const;
  const void* comm_net_token() const;
  int64_t eord_regst_desc_id() const;

  // Serialize
  template<typename StreamT>
  void Serialize(StreamT& out_stream) const {
    out_stream.Write(this, sizeof(ActorMsg));
  }
  template<typename StreamT>
  void Deserialize(StreamT& in_stream) {
    in_stream.Read(this, sizeof(ActorMsg));
  }

 private:
  struct RegstWrapper {
    Regst* regst;
    const void* comm_net_token;
    RegstStatus regst_status;
  };

  int64_t src_actor_id_;
  int64_t dst_actor_id_;
  ActorMsgType msg_type_;
  union {
    ActorCmd actor_cmd_;
    RegstWrapper regst_wrapper_;
    int64_t eord_regst_desc_id_;
  };
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
