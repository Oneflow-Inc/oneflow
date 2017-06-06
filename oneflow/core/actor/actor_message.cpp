#include "oneflow/core/actor/actor_message.h"
#include "oneflow/core/register/remote_register_warpper.h"
#include "oneflow/core/register/local_register_warpper.h"
#include "oneflow/core/common/id_manager.h"
#include "oneflow/core/runtime/runtime_info.h"

namespace oneflow {

ActorMsg::ActorMsg() {
  dst_actor_id_ = std::numeric_limits<uint64_t>::max();
}

ActorMsg ActorMsg::BuildMsgForRegstReader(uint64_t reader_actor_id,
                                          Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.dst_actor_id_ = reader_actor_id;
  if (IDMgr::Singleton().MachineId4ActorId(reader_actor_id) ==
      RuntimeInfo::Singleton().this_machine_id()) {
    msg.regst_warpper_.reset(new LocalRegstWarpper(regst_raw_ptr));
  } else {
    msg.regst_warpper_.reset(new RemoteRegstWarpper(regst_raw_ptr));
  }
  return msg;
}

ActorMsg ActorMsg::BuildMsgForRegstWriter(uint64_t writer_actor_id,
                                          Regst* regst_raw_ptr) {
  ActorMsg msg;
  msg.dst_actor_id_ = writer_actor_id;
  msg.regst_warpper_.reset(new LocalRegstWarpper(regst_raw_ptr));
  return msg;
}

} // namespace oneflow
