#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/machine_context.h"

namespace oneflow {

namespace {

bool IsSoleBlobAndDynamicEmpty(Regst* regst) {
  if (regst == nullptr) { return false; }
  if (regst->GetBlobSize() != 1) { return false; }
  return regst->GetMutSoleBlob()->IsBodyEmpty();
}

}  // namespace

ActorMsg ActorMsg::BuildRegstMsgToConsumer(int64_t producer, int64_t consumer,
                                           Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.src_actor_id_ = producer;
  msg.dst_actor_id_ = consumer;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_wrapper_.regst = regst_raw_ptr;
  if (Global<IDMgr>::Get()->MachineId4ActorId(consumer)
      == Global<MachineCtx>::Get()->this_machine_id()) {
    msg.regst_wrapper_.comm_net_token = nullptr;
  } else {
    msg.regst_wrapper_.comm_net_token = regst_raw_ptr->comm_net_token();
  }
  msg.regst_wrapper_.regst_status = regst_raw_ptr->status();
  msg.regst_wrapper_.is_regst_empty = IsSoleBlobAndDynamicEmpty(regst_raw_ptr);
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
  // you can NOT access the regst ptr when multi nodes, because the address is in another machine
  msg.regst_wrapper_.is_regst_empty = false;
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
  return Global<IDMgr>::Get()->MachineId4ActorId(src_actor_id_);
}

ActorCmd ActorMsg::actor_cmd() const {
  CHECK_EQ(msg_type_, ActorMsgType::kCmdMsg);
  return actor_cmd_;
}

Regst* ActorMsg::regst() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst;
}

int64_t ActorMsg::regst_desc_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  if (Global<IDMgr>::Get()->MachineId4ActorId(src_actor_id_)
      == Global<MachineCtx>::Get()->this_machine_id()) {
    return regst_wrapper_.regst->regst_desc_id();
  } else {
    return regst_wrapper_.regst_status.regst_desc_id;
  }
}

int64_t ActorMsg::piece_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst_status.piece_id;
}

int64_t ActorMsg::act_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.regst_status.act_id;
}

void* ActorMsg::comm_net_token() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.comm_net_token;
}

bool ActorMsg::is_regst_empty() const {
  CHECK_EQ(msg_type_, ActorMsgType::kRegstMsg);
  return regst_wrapper_.is_regst_empty;
}

int64_t ActorMsg::eord_regst_desc_id() const {
  CHECK_EQ(msg_type_, ActorMsgType::kEordMsg);
  return eord_regst_desc_id_;
}

}  // namespace oneflow
