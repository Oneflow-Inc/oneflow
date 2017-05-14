#include "thread/actor_msg_bus.h"
#include "job/id_manager.h"
#include "job/runtime_info.h"

namespace oneflow {

void ActorMsgBus::InsertThrdLocIdMsgQPair(
    uint64_t thrd_loc_id, 
    std::unique_ptr<BlockingChannel<ActorMsg>> msg_queue) {
  CHECK(thrd_loc_id2msg_queue_.emplace(
        thrd_loc_id, std::move(msg_queue)).second);
}

void ActorMsgBus::SendMsg(const ActorMsg& msg) {
  uint64_t dst_machine_id = 
    IDMgr::Singleton().MachineId4ActorId(msg.dst_actor_id);
  if (dst_machine_id == RuntimeInfo::Singleton().machine_id()) {
    uint64_t thrd_loc_id = 
      IDMgr::Singleton().ThrdLocId4ActorId(msg.dst_actor_id);
    thrd_loc_id2msg_queue_.at(thrd_loc_id)->Write(msg);
  } else {
    SendMsg(msg, dst_machine_id);
  }
}

void ActorMsgBus::SendMsg(const ActorMsg& msg, uint64_t dst_machine_id) {
  TODO();
}

}  // namespace oneflow
