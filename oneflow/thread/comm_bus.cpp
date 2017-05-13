#include "thread/comm_bus.h"
#include "job/id_manager.h"

namespace oneflow {

void CommBus::InsertThrdLocIdMsgQPair(
    uint64_t thrd_loc_id, BlockingQueue* msg_queue) {
  CHECK(thrd_loc_id2msg_queue_.emplace(thrd_loc_id, msg_queue).second);
}

void CommBus::SendMsg(const Message& msg) {
  uint64_t dst_machine_id = 
    IDMgr::Singleton().MachineId4ActorId(msg.to_actor_id());
  if (dst_machine_id == this_machine_id) {
    uint64_t thrd_loc_id = 
      IDMgr::Singleton().ThrdLocId4ActorId(msg.to_actor_id());
    thrd_loc_id2msg_queue_.at(thrd_loc_id)->Write(msg);
  } else {
    SendMsg(msg, dst_machine_id);
  }
}

void CommBus::SendMsg(const Message& msg, uint64_t dst_machine_id) {
  TODO();
}

}  // namespace oneflow
