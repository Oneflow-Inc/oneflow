#include "oneflow/core/actor/repeat_backward_compute_actor.h"

namespace oneflow {

void RepeatBackwardCompActor::VirtualCompActorInit(const TaskProto& proto) {
  CHECK_EQ(proto.exec_sequence().exec_node().size(), 1);
  const KernelConf& kernel_conf = proto.exec_sequence().exec_node().Get(0).kernel_conf();
  CHECK(kernel_conf.op_attribute().op_conf().has_repeat_conf());
  const Shape& out_diff_time_shape = Global<RegstMgr>::Get()
                                         ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out_diff"))
                                         .data_regst_time_shape();
  CHECK_GE(out_diff_time_shape.NumAxes(), 1);
  repeat_num_ = out_diff_time_shape.At(out_diff_time_shape.NumAxes() - 1);
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
