#include "oneflow/core/actor/fw_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void FwDataCompActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  expected_model_version_id_ = 0;
  num_of_read_over_ = 0;
  cur_msg_handle_ = &FwDataCompActor::HandleInitDeviceCtx;
}

bool FwDataCompActor::IsReadReady() {
  if (model_regst_ && model_tmp_regst_ && !in_.empty()) {
    // More Effective Distributed ML via a Stale Synchronous Parallel Parameter Server
    uint32_t staleness = JobDesc::Singleton().staleness();
    uint32_t num_of_piece_in_batch = JobDesc::Singleton().num_of_piece_in_batch();
    uint64_t cur_iteration = in_.front()->piece_id() / num_of_piece_in_batch;
    uint64_t stale_version = cur_iteration - staleness;
    return model_regst_->model_version_id() >= stale_version;
  }
  return false;
}

int FwDataCompActor::ProcessMsg(const ActorMsg& msg,
                                const ThreadContext& thread_ctx) {
  return (this->*cur_msg_handle_)(msg, thread_ctx);
}

int FwDataCompActor::HandleInitDeviceCtx(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kInitDeviceCtx);
  if (thread_ctx.cpu_stream) {
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  cur_msg_handle_ = &FwDataCompActor::HandleFwComp;
  return 0;
}

int FwDataCompActor::HandleFwComp(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kOneRegstDescDone);
    num_of_read_over_ += 1;
    if (num_of_read_over_ == 3) {
      cur_msg_handle_ = &FwDataCompActor::HandleFwCompWhenNoReadableRegstMsg;
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()) != 0) {
      std::shared_ptr<RegstWarpper> regst_wp = msg.regst_warpper();
      if (regst_wp->regst_desc_id() == model_tmp_regst_desc_id_) {
        CHECK(!model_tmp_regst_);
        model_tmp_regst_ = regst_wp;
        ready_in_regst_[model_tmp_regst_desc_id_] = regst_wp;
      } else if (regst_wp->regst_desc_id() == model_regst_desc_id_) {
        CHECK_EQ(regst_wp->model_version_id(), expected_model_version_id_);
        if (model_regst_) {
          AsyncSendRegstMsgToProducer(model_regst_);
        }
        model_regst_ = regst_wp;
        ready_in_regst_[model_regst_desc_id_] = regst_wp;
        expected_model_version_id_ += 1;
      } else {
        in_.push(regst_wp);
      }
    }
  }
  TryWardKernelAndSendMsg();
  return 0;
}
int FwDataCompActor::HandleFwCompWhenNoReadableRegstMsg(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()), 0);
  TryWardKernelAndSendMsg();
  if (in_.empty()) {
    AsyncSendRegstMsgToProducer(model_regst_);
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    AsyncSendRegstDescDoneMsgForAllProducedRegstDesc();
    if (total_reading_cnt() == 0) {
      cur_msg_handle_ = nullptr;
      return 1;
    } else {
      cur_msg_handle_ = &FwDataCompActor::HandleWaitUntilReadingCntEqualZero;
      return 0;
    }
  }
  return 0;
}
  
int FwDataCompActor::HandleWaitUntilReadingCntEqualZero(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()), 0);
  if (total_reading_cnt() == 0) {
    cur_msg_handle_ = nullptr;
    return 1;
  }
  return 0;
}

void FwDataCompActor::TryWardKernelAndSendMsg() {
  while (IsReadReady() && IsWriteReady()) {
    CHECK_EQ(in_.front()->piece_id(), expected_piece_id());
    ready_in_regst_[in_.front()->regst_desc_id()] = in_.front();
    uint64_t piece_id = in_.front()->piece_id();
    uint64_t model_version_id = model_regst_->model_version_id();
    AsyncWardKernel(GenDefaultKernelCtx(), 
        [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      Regst* regst = GetCurWriteableRegst(regst_desc_id);
      if (regst == nullptr) {
        return ready_in_regst_.at(regst_desc_id);
      } else {
        return std::make_shared<LocalRegstWarpper> (regst);
      }
    });
    ForEachCurWriteableRegst([piece_id, model_version_id](Regst* regst) {
      regst->set_piece_id(piece_id);
      regst->set_model_version_id(model_version_id);
    });
    AsyncSendReadableRegstMsg();
    AsyncSendRegstMsgToProducer(in_.front());
    in_.pop();
  }
}

REGISTER_ACTOR(kDataCompTask, true, FwDataCompActor);

}  // namespace oneflow
