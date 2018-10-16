#include "oneflow/core/actor/unpack_compute_actor.h"

namespace oneflow {

void UnpackCompActor::VirtualCompActorInit(const TaskProto& proto) {
  int64_t in_diff_regst_desc_id = Name2SoleRegstDescId("in_diff");
  handle_pack_bw_ = in_diff_regst_desc_id != -1;
  if (handle_pack_bw_) {
    const Shape& in_diff_time_shape = Global<RegstMgr>::Get()
                                          ->RegstDesc4RegstDescId(in_diff_regst_desc_id)
                                          .data_regst_time_shape();
    total_unpack_num_ = in_diff_time_shape.At(in_diff_time_shape.NumAxes() - 1);
  } else {
    const Shape& out_time_shape = Global<RegstMgr>::Get()
                                      ->RegstDesc4RegstDescId(Name2SoleRegstDescId("out"))
                                      .data_regst_time_shape();
    total_unpack_num_ = out_time_shape.At(out_time_shape.NumAxes() - 1);
  }
  in_regst_desc_id_ = Name2SoleRegstDescId("in");
  act_num_cnt_ = 0;
  cur_piece_id_ = 0;
  is_in_eord_ = false;
  OF_SET_MSG_HANDLER(&UnpackCompActor::HandlerNormal);
}

void UnpackCompActor::NormalProcessCustomizedEordMsg(const ActorMsg& msg) {
  CHECK_EQ(in_regst_desc_id_, msg.eord_regst_desc_id());
  is_in_eord_ = true;
}

void UnpackCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  if (handle_pack_bw_) {
    AsyncSendRegstMsgToProducer(msg.regst());
  } else {
    in_regsts_.push(msg.regst());
  }
}

bool UnpackCompActor::IsCustomizedReadReady() {
  if (handle_pack_bw_) {
    CHECK(in_regsts_.empty());
    return true;
  }
  return in_regsts_.empty() == false;
}

bool UnpackCompActor::IsCustomizedReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && in_regsts_.empty();
}

void UnpackCompActor::AsyncReturnAllCustomizedReadableRegst() {
  CHECK(is_in_eord_);
  CHECK(in_regsts_.empty());
}

void UnpackCompActor::Act() {
  KernelCtx ctx = GenDefaultKernelCtx();
  std::pair<size_t, size_t> other_val = std::make_pair(act_num_cnt_, total_unpack_num_);
  ctx.other = static_cast<void*>(&other_val);
  AsyncLaunchKernel(ctx, [this](int64_t regst_desc_id) {
    CHECK(handle_pack_bw_ == false);
    CHECK_EQ(in_regst_desc_id_, regst_desc_id);
    return in_regsts_.front();
  });
  act_num_cnt_ += 1;
}

void UnpackCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  HandleProducedNaiveDataRegstToConsumer([this](Regst* regst) {
    regst->set_piece_id(cur_piece_id_);
    return true;
  });
  cur_piece_id_ += 1;
}

void UnpackCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  if (act_num_cnt_ == total_unpack_num_) {
    if (handle_pack_bw_) {
      CHECK(in_regsts_.empty());
    } else {
      CHECK(in_regsts_.empty() == false);
      AsyncSendRegstMsgToProducer(in_regsts_.front());
      in_regsts_.pop();
    }
  }
}

void UnpackCompActor::VirtualAsyncSendNaiveConsumedRegstMsgToProducer() {
  if (act_num_cnt_ == total_unpack_num_) {
    if (handle_pack_bw_) {
      HandleConsumedNaiveDataRegstToProducer([](Regst*) { return true; });
    }
    act_num_cnt_ = 0;
  }
}

REGISTER_ACTOR(TaskType::kUnpackForward, UnpackCompActor);
REGISTER_ACTOR(TaskType::kPackBackward, UnpackCompActor);

}  // namespace oneflow
