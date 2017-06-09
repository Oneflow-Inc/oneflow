#include "oneflow/core/actor/fw_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void FwDataCompActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  expected_model_version_id_ = 0;
}

bool FwDataCompActor::IsReadReady() {
  uint32_t staleness = JobDesc::Singleton().staleness();
  uint32_t num_of_piece_in_batch = JobDesc::Singleton().num_of_piece_in_batch();
  if (model_regst_ != nullptr && model_tmp_regst_ != nullptr && !in_.empty()) {
    if(model_regst_->model_version_id() + staleness - 1 >= 
       in_.front()->piece_id() / num_of_piece_in_batch) {
      return true;
    }
  }
  return false;
}

void FwDataCompActor::ProcessMsg(const ActorMsg& msg,
                                 const ThreadContext& thread_ctx) {
  KernelContext kernel_ctx;
  kernel_ctx.cuda_stream = thread_ctx.compute_cuda_stream;
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    TODO();
  }
  if (TryUpdtStateAsFromRegstReader(msg.regst_warpper()->regst_raw_ptr()) != 0) {
    std::shared_ptr<RegstWarpper> regst_wp = msg.regst_warpper();
    if (regst_wp->regst_desc_id() == model_tmp_regst_desc_id_) {
      CHECK(!model_tmp_regst_);
      model_tmp_regst_ = regst_wp;
      ready_in_regst_[model_tmp_regst_desc_id_] = regst_wp;
    } else if (regst_wp->regst_desc_id() == model_regst_desc_id_) {
      CHECK_EQ(regst_wp->model_version_id(), expected_model_version_id_);
      ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
            model_regst_->producer_actor_id(),
            model_regst_->regst_raw_ptr()));
      model_regst_ = regst_wp;
      ready_in_regst_[model_regst_desc_id_] = regst_wp;
      expected_model_version_id_ += 1;
    } else {
      in_.push(regst_wp);
    }
  }
  while (IsReadReady() && IsWriteReady()) {
    WardKernelAndSendMsg(kernel_ctx);
  }
}

void FwDataCompActor::WardKernelAndSendMsg(const KernelContext& kernel_ctx) {
  CHECK_EQ(in_.front()->piece_id(), expected_piece_id());
  ready_in_regst_[in_.front()->regst_desc_id()] = in_.front();
  uint64_t piece_id = in_.front()->piece_id();
  uint64_t model_version_id = model_regst_->model_version_id();
  WardKernel(kernel_ctx, [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      return ready_in_regst_[regst_desc_id];
    } else {
      return std::make_shared<LocalRegstWarpper> (regst);
    }
  });
  ForEachCurWriteableRegst([piece_id, model_version_id](Regst* regst) {
    regst->set_piece_id(piece_id);
    regst->set_model_version_id(model_version_id);
  });
  CurWriteDone();
  ActorMsgBus::Singleton().SendMsg(ActorMsg::BuildMsgForRegstWriter(
        in_.front()->producer_actor_id(),
        in_.front()->regst_raw_ptr()));
  in_.pop();
}

REGISTER_ACTOR(kDataCompTask, true, FwDataCompActor);

}  // namespace oneflow
