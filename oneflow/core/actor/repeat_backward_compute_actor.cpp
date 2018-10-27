#include "oneflow/core/actor/repeat_backward_compute_actor.h"
#include "oneflow/core/operator/repeat_op.h"

namespace oneflow {

void RepeatBackwardCompActor::VirtualCompActorInit(const TaskProto& proto) {
  CHECK_EQ(proto.exec_sequence().exec_node().size(), 1);
  const KernelConf& kernel_conf = proto.exec_sequence().exec_node().Get(0).kernel_conf();
  CHECK(kernel_conf.op_attribute().op_conf().has_repeat_conf());
  repeat_num_ =
      RepeatOp::GetRepeatNum(kernel_conf.op_attribute().op_conf().repeat_conf(), *parallel_ctx());
  acc_count_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&RepeatBackwardCompActor::HandlerNormal);
}

void RepeatBackwardCompActor::Act() {
  // reset acc_count_ if need
  if (acc_count_ == repeat_num_) { acc_count_ = 0; }
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  kernel_ctx.other = &acc_count_;
  AsyncLaunchKernel(kernel_ctx);
  acc_count_ += 1;
}

void RepeatBackwardCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (acc_count_ == repeat_num_) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      regst->set_piece_id(cur_piece_id_);
      return true;
    });
    cur_piece_id_ += 1;
  }
}

REGISTER_ACTOR(kRepeatBackward, RepeatBackwardCompActor);

}  // namespace oneflow
