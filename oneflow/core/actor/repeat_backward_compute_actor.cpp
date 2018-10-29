#include "oneflow/core/actor/repeat_backward_compute_actor.h"

namespace oneflow {

void RepeatBackwardCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& out_diff_time_shape = Global<RegstMgr>::Get()
                                         ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out_diff"))
                                         .data_regst_time_shape();
  const Shape& in_diff_time_shape = Global<RegstMgr>::Get()
                                        ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in_diff"))
                                        .data_regst_time_shape();
  CHECK_GE(out_diff_time_shape.NumAxes(), 1);
  CHECK_EQ(in_diff_time_shape.NumAxes() + 1, out_diff_time_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, in_diff_time_shape.NumAxes()) {
    CHECK_EQ(in_diff_time_shape.At(i), out_diff_time_shape.At(i));
  }
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
