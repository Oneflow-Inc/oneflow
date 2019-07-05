#include "oneflow/core/actor/instance_stack_compute_actor.h"

namespace oneflow {

void InstanceStackCompActor::VirtualCompActorInit(const TaskProto& proto) {
  int64_t out_diff_regst_desc_id = Name2SoleRegstDescId("out_diff");
  handle_piece_slice_bw_ = out_diff_regst_desc_id != -1;
  if (handle_piece_slice_bw_) {
    const Shape& out_diff_time_shape = Global<RegstMgr>::Get()
                                           ->RegstDesc4RegstDescId(out_diff_regst_desc_id)
                                           .data_regst_time_shape();
    total_stack_num_ = out_diff_time_shape.At(out_diff_time_shape.NumAxes() - 1);
  } else {
    const Shape& in_time_shape = Global<RegstMgr>::Get()
                                     ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                     .data_regst_time_shape();
    total_stack_num_ = in_time_shape.At(in_time_shape.NumAxes() - 1);
  }
  act_num_cnt_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&InstanceStackCompActor::HandlerNormal);
}

void InstanceStackCompActor::Act() {
  KernelCtx ctx = GenDefaultKernelCtx();
  std::pair<size_t, size_t> other_val = std::make_pair(act_num_cnt_, total_stack_num_);
  ctx.other = static_cast<void*>(&other_val);
  AsyncLaunchKernel(ctx);
  act_num_cnt_ += 1;
}

void InstanceStackCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (act_num_cnt_ == total_stack_num_) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      regst->set_piece_id(cur_piece_id_);
      return true;
    });
    cur_piece_id_ += 1;
  }
}

void InstanceStackCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  HandleConsumedNaiveDataRegstToProducer([](Regst*) { return true; });
  if (act_num_cnt_ == total_stack_num_) { act_num_cnt_ = 0; }
}

REGISTER_ACTOR(TaskType::kInstanceStackForward, InstanceStackCompActor);
REGISTER_ACTOR(TaskType::kPieceSliceBackward, InstanceStackCompActor);

}  // namespace oneflow
