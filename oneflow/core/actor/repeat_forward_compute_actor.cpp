#include "oneflow/core/actor/repeat_forward_compute_actor.h"
#include "oneflow/core/operator/repeat_op.h"

namespace oneflow {

void RepeatForwardCompActor::VirtualCompActorInit(const TaskProto& proto) {
  CHECK_EQ(proto.exec_sequence().exec_node().size(), 1);
  const KernelConf& kernel_conf = proto.exec_sequence().exec_node().Get(0).kernel_conf();
  CHECK(kernel_conf.op_attribute().op_conf().has_repeat_conf());
  repeat_num_ =
      RepeatOp::GetRepeatNum(kernel_conf.op_attribute().op_conf().repeat_conf(), *parallel_ctx());
  repeat_count_ = 0;
  cur_piece_id_ = 0;

  const RegstDescProto& out_regst_desc = proto.produced_regst_desc().at("out");
  CHECK(!out_regst_desc.enable_mem_sharing());
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
