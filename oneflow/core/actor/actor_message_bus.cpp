#include "oneflow/core/actor/actor_message_bus.h"
#include "oneflow/core/job/id_manager.h"
#include "oneflow/core/job/machine_context.h"
#include "oneflow/core/thread/thread_manager.h"
#include "oneflow/core/comm_network/comm_network.h"

namespace oneflow {

void ActorMsgBus::SendMsg(const ActorMsg& msg) {
  int64_t dst_machine_id = Global<IDMgr>::Get()->MachineId4ActorId(msg.dst_actor_id());
  if (dst_machine_id == Global<MachineCtx>::Get()->this_machine_id()) {
    SendMsgWithoutCommNet(msg);
  } else {
    Global<CommNet>::Get()->SendActorMsg(dst_machine_id, msg);
  }
}

void ActorMsgBus::SendMsgWithoutCommNet(const ActorMsg& msg) {
  CHECK_EQ(Global<IDMgr>::Get()->MachineId4ActorId(msg.dst_actor_id()),
           Global<MachineCtx>::Get()->this_machine_id());
  int64_t thrd_id = Global<IDMgr>::Get()->ThrdId4ActorId(msg.dst_actor_id());
  Global<ThreadMgr>::Get()->GetThrd(thrd_id)->GetMsgChannelPtr()->Send(msg);
}

}  // namespace oneflow
