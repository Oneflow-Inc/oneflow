#include "oneflow/core/actor/wait_and_send_ids_compute_actor.h"
#include "oneflow/core/job/runtime_context.h"
#include "oneflow/core/record/record.pb.h"

namespace oneflow {

void WaitAndSendIdsCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  wait_and_send_ids_status_.channel_status_ = kChannelStatusSuccess;
  wait_and_send_ids_status_.in_id_ = 0;
  wait_and_send_ids_status_.out_idx_ = 0;
  wait_and_send_ids_status_.out_num_ = 0;
  OF_SET_MSG_HANDLER(&WaitAndSendIdsCompActor::HandlerWaitToStart);
}

void WaitAndSendIdsCompActor::Act() {
  CHECK_LE(wait_and_send_ids_status_.out_idx_, wait_and_send_ids_status_.out_num_);
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &wait_and_send_ids_status_;
  AsyncLaunchKernel(kernel_ctx);
}

void WaitAndSendIdsCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (wait_and_send_ids_status_.channel_status_ == kChannelStatusSuccess) {
    HandleProducedNaiveDataRegstToConsumer();
  }
}

bool WaitAndSendIdsCompActor::IsCustomizedReadReady() const {
  return wait_and_send_ids_status_.channel_status_ == kChannelStatusSuccess;
}

int WaitAndSendIdsCompActor::HandlerWaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  OF_SET_MSG_HANDLER(&WaitAndSendIdsCompActor::HandlerNormal);
  return ProcessMsg(msg);
}

REGISTER_ACTOR(kWaitAndSendIds, WaitAndSendIdsCompActor);

}  // namespace oneflow
