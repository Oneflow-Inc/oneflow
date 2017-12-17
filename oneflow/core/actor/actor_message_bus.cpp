#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/comm_network/comm_network.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/thread/thread_manager.h"

namespace oneflow {

void ActorMsgBus::SendMsg(const ActorMsg& msg) {
  int64_t dst_machine_id =
      IDMgr::Singleton()->MachineId4ActorId(msg.dst_actor_id());
  if (dst_machine_id == MachineCtx::Singleton()->this_machine_id()) {
    int64_t thrd_id = IDMgr::Singleton()->ThrdId4ActorId(msg.dst_actor_id());
    ThreadMgr::Singleton()->GetThrd(thrd_id)->GetMsgChannelPtr()->Send(msg);
  } else {
    CommNet::Singleton()->SendActorMsg(dst_machine_id, msg);
  }
}

}  // namespace oneflow
