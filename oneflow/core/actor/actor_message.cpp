#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/register/local_register_warpper.h"
#include "oneflow/core/register/remote_register_warpper.h"

namespace oneflow {

OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorCmd);
OF_DEFINE_ENUM_TO_OSTREAM_FUNC(ActorMsgType);

ActorMsg::ActorMsg() { dst_actor_id_ = -1; }

ActorMsg ActorMsg::BuildReadableRegstMsg(int64_t reader_actor_id,
                                         Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.dst_actor_id_ = reader_actor_id;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  if (IDMgr::Singleton()->MachineId4ActorId(reader_actor_id)
      == RuntimeCtx::Singleton()->this_machine_id()) {
    msg.regst_warpper_.reset(new LocalRegstWarpper(regst_raw_ptr));
  } else {
    msg.regst_warpper_.reset(new RemoteRegstWarpper(regst_raw_ptr));
  }
  return msg;
}

ActorMsg ActorMsg::BuildRegstMsgToProducer(int64_t writer_actor_id,
                                           Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.dst_actor_id_ = writer_actor_id;
  msg.msg_type_ = ActorMsgType::kRegstMsg;
  msg.regst_warpper_.reset(new LocalRegstWarpper(regst_raw_ptr));
  return msg;
}

}  // namespace oneflow
