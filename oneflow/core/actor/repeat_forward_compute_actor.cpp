#include "oneflow/core/actor/repeat_forward_compute_actor.h"

namespace oneflow {

void RepeatForwardCompActor::VirtualCompActorInit(const TaskProto& proto) {
  CHECK_EQ(proto.exec_sequence().exec_node().size(), 1);
  const KernelConf& kernel_conf = proto.exec_sequence().exec_node().Get(0).kernel_conf();
  CHECK(kernel_conf.op_attribute().op_conf().has_repeat_conf());
  repeat_num_ = kernel_conf.op_attribute().op_conf().repeat_conf().repeat_num();
  OF_SET_MSG_HANDLER(&RepeatForwardCompActor::HandlerNormal);
}

void RepeatForwardCompActor::Act() {
  // reset repeat_count if need
  if (repeat_count_ == repeat_num_) { repeat_count_ = 0; }

  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  AsyncLaunchKernel(kernel_ctx);
  repeat_count_ += 1;
}

void RepeatForwardCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (repeat_count_ == repeat_num_) {
    HandleConsumedNaiveDataRegstToProducer([](Regst* regst) { return true; });
  }
}

void RepeatForwardCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  int64_t in_regst_piece_id = GetNaiveCurReadable("in")->piece_id();

  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(in_regst_piece_id);
    return true;
  });
}

REGISTER_ACTOR(kRepeatForward, RepeatForwardCompActor);

}  // namespace oneflow
