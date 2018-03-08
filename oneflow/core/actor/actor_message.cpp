#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorCmd);
OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorMsgType);

ActorMsg ActorMsg::BuildRegstMsgToConsumer(int64_t producer, int64_t consumer,
                                           Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.src_actor_id_ = producer;
  msg.dst_actor_id_ = consumer;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_wrapper_.regst = regst_raw_ptr;
  if (IDMgr::Singleton()->MachineId4ActorId(consumer)
      == MachineCtx::Singleton()->this_machine_id()) {
    msg.regst_wrapper_.comm_net_token = nullptr;
  } else {
    msg.regst_wrapper_.comm_net_token =
        regst_raw_ptr->packed_blob()->comm_net_token();
    msg.regst_wrapper_.regst_status = regst_raw_ptr->status();
  }
  return msg;
}

ActorMsg ActorMsg::BuildRegstMsgToProducer(int64_t consumer, int64_t producer,
                                           Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.src_actor_id_ = consumer;
  msg.dst_actor_id_ = producer;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_wrapper_.regst = regst_raw_ptr;
  msg.regst_wrapper_.comm_net_token = nullptr;
  return msg;
}

ActorMsg ActorMsg::BuildEordMsg(int64_t consumer, int64_t regst_desc_id) {
  ActorMsg msg;
  msg.src_actor_id_ = -1;
  msg.dst_actor_id_ = consumer;
  msg.msg_type_ = ActorMsgType::kEordMsg;
  msg.eord_regst_desc_id_ = regst_desc_id;
  return msg;
}

ActorMsg ActorMsg::BuildCommandMsg(int64_t dst_actor_id, ActorCmd cmd) {
  ActorMsg msg;
  msg.src_actor_id_ = -1;
  msg.dst_actor_id_ = dst_actor_id;
  msg.msg_type_ = ActorMsgType::kCmdMsg;
  msg.actor_cmd_ = cmd;
  return msg;
}

int64_t ActorMsg::SrcMachineId() const {
  return IDMgr::Singleton()->MachineId4ActorId(src_actor_id_);
}

ActorCmd ActorMsg::actor_cmd() const {
  CHECK_EQ(msg_type_, ActorMsgType::kCmdMsg);
  return actor_cmd_;
}

Regst* ActorMsg::regst() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst;
}

int64_t ActorMsg::piece_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst_status.piece_id;
}

int64_t ActorMsg::act_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst_status.act_id;
}

const void* ActorMsg::comm_net_token() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.comm_net_token;
}

int64_t ActorMsg::eord_regst_desc_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kEordMsg);
  return eord_regst_desc_id_;
}

}  // namespace oneflow
