#include "oneflow/core/actor/unpack_compute_actor.h"

namespace oneflow {

void UnpackCompActor::VirtualCompActorInit(const TaskProto& proto) {
  int64_t out_diff_regst_desc_id = Name2SoleRegstDescId("out_diff");
  handle_pack_bw_ = out_diff_regst_desc_id != -1;
  if (handle_pack_bw_) {
    const Shape& out_diff_time_shape = Global<RegstMgr>::Get()
                                           ->RegstDesc4RegstDescId(out_diff_regst_desc_id)
                                           .data_regst_time_shape();
    total_unpack_num_ = out_diff_time_shape.At(out_diff_time_shape.NumAxes() - 1);
  } else {
    const Shape& out_time_shape = Global<RegstMgr>::Get()
                                      ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                      .data_regst_time_shape();
    total_unpack_num_ = out_time_shape.At(out_time_shape.NumAxes() - 1);
  }
  act_num_cnt_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&UnpackCompActor::HandlerNormal);
}

void UnpackCompActor::Act() {
  KernelCtx ctx = GenDefaultKernelCtx();
  std::pair<size_t, size_t> other_val = std::make_pair(act_num_cnt_, total_unpack_num_);
  ctx.other = static_cast<void*>(&other_val);
  AsyncLaunchKernel(ctx);
  act_num_cnt_ += 1;
}

void UnpackCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
  cur_piece_id_ += 1;
}

void UnpackCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (handle_pack_bw_) {
    if (act_num_cnt_ == total_unpack_num_) {
      HandleConsumedNaiveDataRegstToProducer([](Regst*) { return true; });
      act_num_cnt_ = 0;
    }
  } else {
    int64_t in_regst_desc_id = Name2SoleRegstDescId("in");
    HandleConsumedNaiveDataRegstToProducer(
        [in_regst_desc_id](Regst* regst) { return regst->regst_desc_id() == in_regst_desc_id; });
    if (act_num_cnt_ == total_unpack_num_) {
      HandleConsumedNaiveDataRegstToProducer([](Regst*) { return true; });
      act_num_cnt_ = 0;
    }
  }
}

REGISTER_ACTOR(TaskType::kUnpackForward, UnpackCompActor);
REGISTER_ACTOR(TaskType::kPieceSliceForward, UnpackCompActor);
REGISTER_ACTOR(TaskType::kInstanceStackBackward, UnpackCompActor);

}  // namespace oneflow
