#include "oneflow/core/actor/repeat_backward_compute_actor.h"
#include "repeat_backward_compute_actor.h"

namespace oneflow {

void RepeatBackwardCompActor::VirtualCompActorInit(const TaskProto& proto) {
  CHECK_EQ(proto.exec_sequence().exec_node().size(), 1);
  const KernelConf& kernel_conf = proto.exec_sequence().exec_node().Get(0).kernel_conf();
  CHECK(kernel_conf.op_attribute().op_conf().has_repeat_conf());
  repeat_num_ = kernel_conf.op_attribute().op_conf().repeat_conf().repeat_num();
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
    Regst* out_diff_regst = GetNaiveCurReadable("out_diff");
    int64_t piece_id = out_diff_regst->piece_id();
    HandleProducedNaiveDataRegstToConsumer([piece_id](Regst* regst) {
      regst->set_piece_id(piece_id);
      return true;
    });
  }
}

bool RepeatBackwardCompActor::ProducedCtrlRegstValid(int64_t regst_desc_id) const {
  return acc_count_ == repeat_num_;
}
REGISTER_ACTOR(kRepeatBackward, RepeatBackwardCompActor);

}  // namespace oneflow
