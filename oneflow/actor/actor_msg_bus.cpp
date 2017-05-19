#include "actor/actor_msg_bus.h"
#include "common/id_manager.h"
#include "runtime/runtime_info.h"
#include "actor/thread_manager.h"

namespace oneflow {

void ActorMsgBus::SendMsg(const ActorMsg& msg) {
  uint64_t dst_machine_id =
    IDMgr::Singleton().MachineId4ActorId(msg.dst_actor_id());
  if (dst_machine_id == RuntimeInfo::Singleton().this_machine_id()) {
    uint64_t thrd_loc_id =
      IDMgr::Singleton().ThrdLocId4ActorId(msg.dst_actor_id());
    ThreadMgr::Singleton().GetMsgChanFromThrdLocId(thrd_loc_id)->Send(msg);
  } else {
    TODO();
  }
}

}  // namespace oneflow
