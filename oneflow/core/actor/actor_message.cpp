#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/runtime_context.h"

namespace oneflow {

OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorCmd);
OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorMsgType);

ActorMsg::ActorMsg() {
  dst_actor_id_ = -1;
  regst_ = nullptr;
}

ActorMsg ActorMsg::BuildReadableRegstMsg(int64_t writer_actor_id,
                                         int64_t reader_actor_id,
                                         Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.src_actor_id_ = writer_actor_id;
  msg.dst_actor_id_ = reader_actor_id;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_ = regst_raw_ptr;
  msg.piece_id_ = regst_raw_ptr->piece_id();
  // if (IDMgr::Singleton()->MachineId4ActorId(reader_actor_id)
  //    == RuntimeCtx::Singleton()->this_machine_id()) {
  //  msg.regst_ = regst_raw_ptr;
  //} else {
  //  TODO();
  //}
  return msg;
}

ActorMsg ActorMsg::BuildRegstMsgToProducer(int64_t writer_actor_id,
                                           int64_t reader_actor_id,
                                           Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.src_actor_id_ = reader_actor_id;
  msg.dst_actor_id_ = writer_actor_id;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_ = regst_raw_ptr;
  return msg;
}

ActorMsg ActorMsg::BuildRegstMsgToProducer(int64_t writer_actor_id,
                                           int64_t reader_actor_id,
                                           Regst* regst_raw_ptr,
                                           int64_t piece_id) {
  ActorMsg msg;
  msg.src_actor_id_ = reader_actor_id;
  msg.dst_actor_id_ = writer_actor_id;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_ = regst_raw_ptr;
  msg.piece_id_ = piece_id;
  return msg;
}

}  // namespace oneflow
