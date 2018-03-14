#include "oneflow/core/actor/record_load_actor.h"

namespace oneflow {

static const int32_t record_load_regst_num = 2;

void RecordLoadActor::Init(const TaskProto& task_proto, const ThreadCtx&) {
  set_actor_id(task_proto.task_id());
  consumers_actor_ids_ = PbRf2StdVec(task_proto.related_decode_task_ids());
  data_path_ = task_proto.data_path();
  record_type_ = task_proto.record_type();
  piece_id_ = 0;
  is_eord_ = true;
  OF_SET_MSG_HANDLER(&RecordLoadActor::HandlerWaitToStart);
}

int RecordLoadActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  ActUntilFail();
  OF_SET_MSG_HANDLER(&RecordLoadActor::HandlerNormal);
  return 0;
}

int RecordLoadActor::HandlerNormal(const ActorMsg& msg) {
  CHECK_EQ(msg.msg_type(), ActorMsgType::kRegstMsg);
  TryUpdtStateAsProducedRegst(msg.regst());
  ActUntilFail();
  if (produced_regsts_.size() == 0) {
    set_msg_handler(static_cast<MsgHandler>(nullptr));
    return 1;
  }
  return 0;
}

void RecordLoadActor::TryUpdtStateAsProducedRegst(Regst* regst) { TODO(); }

void RecordLoadActor::ActUntilFail() {
  while (produced_regsts_.size() < record_load_regst_num && is_eord_) { Act(); }
}

void RecordLoadActor::Act() { TODO(); }

REGISTER_ACTOR(kRecordLoad, RecordLoadActor);

}  // namespace oneflow
