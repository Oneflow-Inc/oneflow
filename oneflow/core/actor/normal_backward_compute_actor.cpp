#include "oneflow/core/actor/normal_backward_compute_actor.h"

namespace oneflow {

void NormalBackwardCompActor::VirtualCompActorInit(const TaskProto& task_proto) {
  any_out_diff_regst_desc_id_ = Name2RegstDescIds("out_diff").front();
  const Shape& out_diff_time_shape = Global<RegstMgr>::Get()
                                         ->RegstDesc4RegstDescId(any_out_diff_regst_desc_id_)
                                         .data_regst_time_shape();
  actual_num_of_piece_in_batch_ = out_diff_time_shape.Count(1);

  model_regst_desc_id_ = Name2SoleRegstDescId("model");
  const_model_regst_desc_id_ = Name2SoleRegstDescId("const_model");
  const_model_regst_ = nullptr;
  const_buf_regst_desc_id_ = Name2SoleRegstDescId("const_buf");
  const_buf_regst_ = nullptr;
  OF_SET_MSG_HANDLER(&NormalBackwardCompActor::HandlerNormal);
}

void NormalBackwardCompActor::ForEachCurCustomizedReadableRegst(
    std::function<void(const Regst*)> handler) const {
  if (model_regst_desc_id_ != -1) {
    CHECK_EQ(model_regst_queue_.empty(), false);
    handler(model_regst_queue_.front());
  }
  if (const_model_regst_desc_id_ != -1) {
    CHECK(const_model_regst_);
    handler(const_model_regst_);
  }
  if (const_buf_regst_desc_id_ != -1) {
    CHECK(const_buf_regst_);
    handler(const_buf_regst_);
  }
}

void NormalBackwardCompActor::NormalProcessNaiveReadableDataRegstMsg(const std::deque<Regst*>& rq) {
  if (rq.size() == 1 && rq.front()->regst_desc_id() == any_out_diff_regst_desc_id_) {
    AsyncReturnModelRegstUntilModelVersionIdEqual(
        GetModelVersionIdFromPieceId(rq.front()->piece_id(), actual_num_of_piece_in_batch_));
  }
}

void NormalBackwardCompActor::NormalProcessCustomizedReadableRegstMsg(const ActorMsg& msg) {
  if (msg.regst()->regst_desc_id() == model_regst_desc_id_) {
    model_regst_queue_.push(msg.regst());
  } else if (msg.regst()->regst_desc_id() == const_model_regst_desc_id_) {
    CHECK(const_model_regst_ == nullptr);
    const_model_regst_ = msg.regst();
  } else if (msg.regst()->regst_desc_id() == const_buf_regst_desc_id_) {
    CHECK(const_buf_regst_ == nullptr);
    const_buf_regst_ = msg.regst();
  } else {
    UNIMPLEMENTED() << msg.regst()->regst_desc_id();
  }
}

void NormalBackwardCompActor::Act() {
  AsyncLaunchKernel(GenDefaultKernelCtx(), [this](int64_t regst_desc_id) -> Regst* {
    if (regst_desc_id == model_regst_desc_id_) {
      return model_regst_queue_.front();
    } else if (regst_desc_id == const_model_regst_desc_id_) {
      return const_model_regst_;
    } else if (regst_desc_id == const_buf_regst_desc_id_) {
      return const_buf_regst_;
    } else {
      return nullptr;
    }
  });
}

void NormalBackwardCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  int64_t piece_id = GetNaiveCurReadable(any_out_diff_regst_desc_id_)->piece_id();
  HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
    regst->set_piece_id(piece_id);
    return true;
  });
}

void NormalBackwardCompActor::AsyncSendCustomizedConsumedRegstMsgToProducer() {
  int64_t piece_id = GetNaiveCurReadable(any_out_diff_regst_desc_id_)->piece_id();
  AsyncReturnModelRegstUntilLastPieceIdGreaterThan(piece_id);
}

bool NormalBackwardCompActor::IsCustomizedReadReady() const {
  if (model_regst_desc_id_ != -1) {
    if (model_regst_queue_.empty()) { return false; }
    int64_t expected_model_vid =
        GetModelVersionIdFromPieceId(GetNaiveCurReadable(any_out_diff_regst_desc_id_)->piece_id(),
                                     actual_num_of_piece_in_batch_);
    CHECK_EQ(expected_model_vid, model_regst_queue_.front()->model_version_id());
  }
  if (const_model_regst_desc_id_ != -1 && const_model_regst_ == nullptr) { return false; }
  if (const_buf_regst_desc_id_ != -1 && const_buf_regst_ == nullptr) { return false; }
  return true;
}

void NormalBackwardCompActor::AsyncReturnAllCustomizedReadableRegst() {
  while (model_regst_queue_.empty() == false) {
    AsyncSendRegstMsgToProducer(model_regst_queue_.front());
    model_regst_queue_.pop();
  }
  if (const_model_regst_) {
    AsyncSendRegstMsgToProducer(const_model_regst_);
    const_model_regst_ = nullptr;
  }
  if (const_buf_regst_) {
    AsyncSendRegstMsgToProducer(const_buf_regst_);
    const_buf_regst_ = nullptr;
  }
}

void NormalBackwardCompActor::AsyncReturnModelRegstUntilModelVersionIdEqual(int64_t model_vid) {
  if (model_regst_desc_id_ == -1) { return; }
  while (!model_regst_queue_.empty()
         && model_regst_queue_.front()->model_version_id() < model_vid) {
    AsyncSendRegstMsgToProducer(model_regst_queue_.front());
    model_regst_queue_.pop();
  }
  if (!model_regst_queue_.empty()) {
    CHECK_EQ(model_regst_queue_.front()->model_version_id(), model_vid);
  }
}

void NormalBackwardCompActor::AsyncReturnModelRegstUntilLastPieceIdGreaterThan(int64_t piece_id) {
  if (model_regst_desc_id_ == -1) { return; }
  while (model_regst_queue_.empty() == false) {
    int64_t model_vid = model_regst_queue_.front()->model_version_id();
    int64_t last_piece_id =
        GetLastPieceIdForModelVersionId(model_vid, actual_num_of_piece_in_batch_);
    if (last_piece_id > piece_id) { return; }
    AsyncSendRegstMsgToProducer(model_regst_queue_.front());
    model_regst_queue_.pop();
  }
}

REGISTER_ACTOR(TaskType::kNormalBackward, NormalBackwardCompActor);

}  // namespace oneflow
