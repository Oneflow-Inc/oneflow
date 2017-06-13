#include "oneflow/core/actor/bp_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

// need review
void BpDataCompActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  activation_regst_desc_id_ = RegstDescId4Name("activation");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");
  expected_model_version_id_ = 0;
}

void BpDataCompActor::ProcessMsg(const ActorMsg& msg,
                                 const ThreadContext& thread_ctx) {
  KernelContext kernel_ctx;
  kernel_ctx.cuda_stream = thread_ctx.compute_cuda_stream;
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    TODO();
  }
  if (TryUpdtStateAsFromRegstReader(msg.regst_warpper()->regst_raw_ptr()) != 0) {
    std::shared_ptr<RegstWarpper> regst_wp = msg.regst_warpper();
    if (regst_wp->regst_desc_id() == model_tmp_regst_desc_id_) {
      CHECK(read_in_.find(model_tmp_regst_desc_id_) != read_in_.end());
    } else if (regst_wp->regst_desc_id() == model_regst_desc_id_) {
      CHECK_EQ(regst_wp->model_version_id(), expected_model_version_id_);
      expected_model_version_id_ += 1;
    }
    read_in_[model_tmp_regst_desc_id_].push(regst_wp);
  }
  while (read_in_.size() == 6 && IsWriteReady()) {
    WardKernelAndSendMsg(kernel_ctx);
  }
}

void BpDataCompActor::WardKernelAndSendMsg(const KernelContext& kernel_ctx) {
  uint64_t cur_model = read_in_.at(model_regst_desc_id_).front()->model_version_id();
  uint64_t piece_id = expected_piece_id();
  CHECK_EQ(cur_model, read_in_.at(activation_regst_desc_id_).front()->model_version_id());
  CHECK_EQ(cur_model, read_in_.at(data_tmp_regst_desc_id_).front()->model_version_id());
  for (const auto& pair : read_in_) {
    if (pair.first != model_regst_desc_id_ && pair.first != model_tmp_regst_desc_id_) {
      CHECK_EQ(pair.second.front()->piece_id(), piece_id);
    }
  }
  WardKernel(kernel_ctx, [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      return read_in_.at(regst_desc_id).front();
    } else {
      return std::make_shared<LocalRegstWarpper> (regst);
    }
  });
  ForEachCurWriteableRegst([piece_id](Regst* regst) {
    regst->set_piece_id(piece_id);
  });
  CurWriteDone();
  for (auto& pair : read_in_) {
    if (pair.first != model_regst_desc_id_ && pair.first != model_tmp_regst_desc_id_) {
      ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
            pair.second.front()->producer_actor_id(),
            pair.second.front()->regst_raw_ptr()));
      pair.second.pop();
    }
  }
  if (!read_in_.at(activation_regst_desc_id_).empty()) {
    if (cur_model != read_in_.at(activation_regst_desc_id_).front()->model_version_id()) {
      ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
            read_in_.at(model_regst_desc_id_).front()->producer_actor_id(),
            read_in_.at(model_regst_desc_id_).front()->regst_raw_ptr()));
      read_in_.at(model_regst_desc_id_).pop();
    }
  }
}

REGISTER_ACTOR(kDataCompTask, false, BpDataCompActor);

}  // namespace oneflow
