#include "oneflow/core/actor/piece_slice_compute_actor.h"

namespace oneflow {

void PieceSliceCompActor::VirtualCompActorInit(const TaskProto& proto) {
  int64_t in_diff_regst_desc_id = Name2SoleRegstDescId("in_diff");
  handle_instance_stack_bw_ = in_diff_regst_desc_id != -1;
  if (handle_instance_stack_bw_) {
    const Shape& in_diff_time_shape = Global<RegstMgr>::Get()
                                          ->RegstDesc4RegstDescId(in_diff_regst_desc_id)
                                          .data_regst_time_shape();
    total_slice_num_ = in_diff_time_shape.At(in_diff_time_shape.NumAxes() - 1);
  } else {
    const Shape& out_time_shape = Global<RegstMgr>::Get()
                                      ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                      .data_regst_time_shape();
    total_slice_num_ = out_time_shape.At(out_time_shape.NumAxes() - 1);
  }
  act_num_cnt_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&PieceSliceCompActor::HandlerNormal);
}

void PieceSliceCompActor::Act() {
  KernelCtx ctx = GenDefaultKernelCtx();
  std::pair<size_t, size_t> other_val = std::make_pair(act_num_cnt_, total_slice_num_);
  ctx.other = static_cast<void*>(&other_val);
  AsyncLaunchKernel(ctx);
  act_num_cnt_ += 1;
}

void PieceSliceCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
  cur_piece_id_ += 1;
}

void PieceSliceCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (act_num_cnt_ == total_slice_num_) {
    HandleConsumedNaiveDataRegstToProducer([](Regst*) { return true; });
    act_num_cnt_ = 0;
  }
}

REGISTER_ACTOR(TaskType::kPieceSliceForward, PieceSliceCompActor);
REGISTER_ACTOR(TaskType::kInstanceStackBackward, PieceSliceCompActor);

}  // namespace oneflow
