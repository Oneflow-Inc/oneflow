#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorCmd);
OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorMsgType);

ActorMsg::ActorMsg() {
  src_actor_id_ = -1;
  dst_actor_id_ = -1;
  regst_wrapper_.regst = nullptr;
  regst_wrapper_.comm_net_token = nullptr;
  regst_wrapper_.piece_id = -1;
}

ActorMsg ActorMsg::BuildRegstMsgToConsumer(int64_t producer, int64_t consumer,
                                           Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.src_actor_id_ = producer;
  msg.dst_actor_id_ = consumer;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_wrapper_.regst = regst_raw_ptr;
  if (IDMgr::Singleton()->MachineId4ActorId(consumer)
      == RuntimeCtx::Singleton()->this_machine_id()) {
    msg.regst_wrapper_.comm_net_token = nullptr;
    msg.regst_wrapper_.piece_id = -1;
  } else {
    msg.regst_wrapper_.comm_net_token =
        regst_raw_ptr->packed_blob()->comm_net_token();
    msg.regst_wrapper_.piece_id = regst_raw_ptr->piece_id();
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
  msg.regst_wrapper_.piece_id = -1;
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
  return regst_wrapper_.piece_id;
}

const void* ActorMsg::comm_net_token() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.comm_net_token;
}

}  // namespace oneflow
