#include "oneflow/core/actor/fw_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void FwDataCompActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  expected_model_version_id_ = 0;
}

bool FwDataCompActor::IsReadReady(uint32_t staleness,
                                  uint32_t num_of_piece_in_batch) {
  if (model_regst_ != nullptr && model_tmp_regst_ != nullptr && !in_.empty()) {
    uint64_t piece_id = in_.front()->piece_id();
    if(model_regst_->model_version_id() + staleness - 1 >= 
       piece_id / num_of_piece_in_batch) {
      return true;
    }
  }
  return false;
}

void FwDataCompActor::ProcessMsg(const ActorMsg& msg,
                                 const ThreadContext& thread_ctx) {
  KernelContext kernel_ctx;
  uint32_t staleness = JobDesc::Singleton().staleness();
  uint32_t num_of_piece_in_batch = JobDesc::Singleton().num_of_piece_in_batch();
  if (TryOneReadDone(msg.regst_warpper()->regst_raw_ptr()) != 0) {
    std::shared_ptr<RegstWarpper> regst_wp = msg.regst_warpper();
    if (regst_wp->regst_desc_id() == model_tmp_regst_desc_id_) {
      model_tmp_regst_ = regst_wp;
    } else if (regst_wp->regst_desc_id() == model_regst_desc_id_) {
      CHECK_EQ(regst_wp->model_version_id(), expected_model_version_id_);
      model_regst_ = regst_wp;
    } else {
      in_.push(regst_wp);
    }
  }
  while (IsReadReady(staleness, num_of_piece_in_batch) && IsWriteReady()) {
    WardKernelAndSendMsg(kernel_ctx);
  }
}

void FwDataCompActor::WardKernelAndSendMsg(const KernelContext& kernel_ctx) {
  WardKernel(kernel_ctx, [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      return ready_in_regst_.front().second->at(regst_desc_id);
    } else {
      return std::make_shared<LocalRegstWarpper> (regst);
    }
  });
  ForEachCurWriteableRegst([piece_id](Regst* regst) {
    regst->set_piece_id(piece_id);
  });
  CurWriteDone();
}

REGISTER_ACTOR(kDataCompTask, true, FwDataCompActor);

}  // namespace oneflow
