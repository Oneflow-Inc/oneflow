#include "oneflow/core/actor/pack_compute_actor.h"

namespace oneflow {

void PackCompActor::VirtualCompActorInit(const TaskProto& proto) {
  int64_t out_diff_regst_desc_id = Name2SoleRegstDescId("out_diff");
  handle_unpack_bw_ = out_diff_regst_desc_id != -1;
  if (handle_unpack_bw_) {
    const Shape& out_diff_time_shape = Global<RegstMgr>::Get()
                                           ->RegstDesc4RegstDescId(out_diff_regst_desc_id)
                                           .data_regst_time_shape();
    total_pack_num_ = out_diff_time_shape.At(out_diff_time_shape.NumAxes() - 1);
  } else {
    const Shape& in_time_shape = Global<RegstMgr>::Get()
                                     ->RegstDesc4RegstDescId(Name2SoleRegstDescId("in"))
                                     .data_regst_time_shape();
    total_pack_num_ = in_time_shape.At(in_time_shape.NumAxes() - 1);
  }
  act_num_cnt_ = 0;
  cur_piece_id_ = 0;
  OF_SET_MSG_HANDLER(&PackCompActor::HandlerNormal);
}

void PackCompActor::Act() {
  KernelCtx ctx = GenDefaultKernelCtx();
  std::pair<size_t, size_t> other_val = std::make_pair(act_num_cnt_, total_pack_num_);
  ctx.other = static_cast<void*>(&other_val);
  AsyncLaunchKernel(ctx);
  act_num_cnt_ += 1;
}

void PackCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (act_num_cnt_ == total_pack_num_) {
    HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
      regst->set_piece_id(cur_piece_id_);
      return true;
    });
    cur_piece_id_ += 1;
  }
}

void PackCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (handle_unpack_bw_ == false) {
    HandleConsumedNaiveDataRegstToProducer([](Regst*) { return true; });
  } else {
    int64_t in_regst_desc_id = Name2SoleRegstDescId("in");
    HandleConsumedNaiveDataRegstToProducer(
        [in_regst_desc_id](Regst* regst) { return regst->regst_desc_id() != in_regst_desc_id; });
    if (act_num_cnt_ == total_pack_num_) {
      AsyncSendRegstMsgToProducer(GetNaiveCurReadable(in_regst_desc_id));
    }
  }
  if (act_num_cnt_ == total_pack_num_) { act_num_cnt_ = 0; }
}

REGISTER_ACTOR(TaskType::kPackForward, PackCompActor);

}  // namespace oneflow
