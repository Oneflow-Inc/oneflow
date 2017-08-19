#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

void ActorMsgBus::SendMsg(const ActorMsg& msg) {
  int64_t dst_machine_id =
      IDMgr::Singleton()->MachineId4ActorId(msg.dst_actor_id());
  if (dst_machine_id == RuntimeCtx::Singleton()->this_machine_id()) {
    int64_t thrd_loc_id =
        IDMgr::Singleton()->ThrdLocId4ActorId(msg.dst_actor_id());
    ThreadMgr::Singleton()->GetThrd(thrd_loc_id)->GetMsgChannelPtr()->Send(msg);
  } else {
    Network* net = GetRdmaInstance();
    NetworkMessage net_msg;
    net_msg.actor_msg = msg;
    net_msg.src_machine_id = RuntimeCtx::Singleton()->this_machine_id();
    net_msg.dst_machine_id = dst_machine_id;
    net_msg.piece_id = msg.regst()->piece_id();
    net->SendMsg(net_msg);
  }
}

}  // namespace oneflow
