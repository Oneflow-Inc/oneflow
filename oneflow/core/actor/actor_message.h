#ifndef ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
#define ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_

#include "oneflow/core/common/util.h"
#include "oneflow/core/register/register_wrapper.h"

namespace oneflow {

enum class ActorCmd {
  kInitializeModel = 0,  // MdUpdt Actor
  kSendInitialModel,     // MdUpdt Actor
  kEORD,                 // End Of Register Desc, All Actor except Source Actor
  kStart,                // Source Actor
  kStopThread
};

OF_DECLARE_ENUM_TO_OSTREAM_FUNC(ActorCmd);

enum class ActorMsgType { kRegstMsg = 0, kCmdMsg };

OF_DECLARE_ENUM_TO_OSTREAM_FUNC(ActorMsgType);

class ActorMsg final {
 public:
  // OF_DISALLOW_COPY_AND_MOVE(ActorMsg);
  ActorMsg();
  ~ActorMsg() = default;

  static ActorMsg BuildReadableRegstMsg(int64_t reader_actor_id, Regst*);
  static ActorMsg BuildRegstMsgToProducer(int64_t writer_actor_id, Regst*);

  // Getters
  int64_t dst_actor_id() const { return dst_actor_id_; }
  ActorMsgType msg_type() const { return msg_type_; }
  std::shared_ptr<RegstWrapper> regst_wrapper() const {
    CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
    return regst_wrapper_;
  }
  ActorCmd actor_cmd() const {
    CHECK_EQ(msg_type_, ActorMsgType::kCmdMsg);
    return actor_cmd_;
  }

  // Setters
  void set_dst_actor_id(int64_t val) { dst_actor_id_ = val; }
  void set_regst_wrapper(std::shared_ptr<RegstWrapper> val) {
    msg_type_ = ActorMsgType::kRegstMsg;
    regst_wrapper_ = val;
  }
  void set_actor_cmd(ActorCmd val) {
    msg_type_ = ActorMsgType::kCmdMsg;
    actor_cmd_ = val;
  }

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
  int64_t dst_actor_id_;
  ActorMsgType msg_type_;

  std::shared_ptr<RegstWrapper> regst_wrapper_;
  ActorCmd actor_cmd_;
  int64_t piece_id_;
  int64_t model_version_id_;
};

}  // namespace oneflow

#endif  // ONEFLOW_CORE_ACTOR_ACTOR_MESSAGE_H_
