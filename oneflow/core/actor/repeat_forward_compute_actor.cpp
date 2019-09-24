#include "oneflow/core/actor/repeat_forward_compute_actor.h"

namespace oneflow {

void RepeatForwardCompActor::VirtualCompActorInit(const TaskProto& proto) {
  const Shape& in_time_shape = Global<RegstMgr>::Get()
                                   ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                   .data_regst_time_shape();
  const Shape& out_time_shape = Global<RegstMgr>::Get()
                                    ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                    .data_regst_time_shape();
  CHECK_GE(out_time_shape.NumAxes(), 1);
  CHECK_EQ(in_time_shape.NumAxes() + 1, out_time_shape.NumAxes());
  FOR_RANGE(int64_t, i, 0, in_time_shape.NumAxes()) {
    CHECK_EQ(in_time_shape.At(i), out_time_shape.At(i));
  }
  repeat_num_ = out_time_shape.At(out_time_shape.NumAxes() - 1);
  repeat_count_ = 0;
  cur_piece_id_ = 0;

  const RegstDescProto& out_regst_desc = proto.produced_regst_desc().at("out");
  CHECK(!out_regst_desc.enable_reuse_mem());
  CHECK_EQ(out_regst_desc.register_num(), 1);
  OF_SET_MSG_HANDLER(&RepeatForwardCompActor::HandlerNormal);
}

void RepeatForwardCompActor::Act() {
  // reset repeat_count if need
  if (repeat_count_ == repeat_num_) { repeat_count_ = 0; }

  if (repeat_count_ == 0) {
    KernelCtx kernel_ctx = GenDefaultKernelCtx();
    AsyncLaunchKernel(kernel_ctx);
  }

  repeat_count_ += 1;
}

void RepeatForwardCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (repeat_count_ == repeat_num_) {
    HandleConsumedNaiveDataRegstToProducer([](Regst* regst) { return true; });
  }
}

void RepeatForwardCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
  cur_piece_id_ += 1;
}

REGISTER_ACTOR(kRepeatForward, RepeatForwardCompActor);

}  // namespace oneflow
