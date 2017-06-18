#include "oneflow/core/actor/bp_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_warpper.h"

namespace oneflow {

void BpDataCompActor::Init(const TaskProto& task_proto) {
  Actor::Init(task_proto);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  activation_regst_desc_id_ = RegstDescId4Name("activation");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");
  expected_model_version_id_ = 0;
  num_of_read_empty_ = 6;
  num_of_read_done_ = 0;
  cur_msg_handle_ = &BpDataCompActor::HandleInitDeviceCtx;
}

bool BpDataCompActor::IsReadReady() {
  if (num_of_read_empty_) {
    return false;
  }
  if (read_regst_.at(model_regst_desc_id_).front()->model_version_id() != 
      read_regst_.at(activation_regst_desc_id_).front()->model_version_id()) {
    AsyncSendRegstMsgToProducer(read_regst_.at(model_regst_desc_id_).front());
    read_regst_.at(model_regst_desc_id_).pop();
    num_of_read_empty_ += read_regst_.at(model_regst_desc_id_).empty();
  }
  return !num_of_read_empty_;
}

int BpDataCompActor::ProcessMsg(const ActorMsg& msg,
                                const ThreadContext& thread_ctx) {
  return (this->*cur_msg_handle_)(msg, thread_ctx);
}

int BpDataCompActor::HandleInitDeviceCtx(
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
  cur_msg_handle_ = &BpDataCompActor::HandleBpComp;
  return 0;
}

int BpDataCompActor::HandleBpComp(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kOneRegstDescDone);
    num_of_read_done_ += 1;
    if (num_of_read_done_ == 6) {
      cur_msg_handle_ = &BpDataCompActor::HandleBpCompWhenNoReadableRegstMsg;
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()) != 0) {
      std::shared_ptr<RegstWarpper> regst_wp = msg.regst_warpper();
      if (regst_wp->regst_desc_id() == model_tmp_regst_desc_id_) {
        CHECK(read_regst_.find(model_tmp_regst_desc_id_) == read_regst_.end());
      } else if (regst_wp->regst_desc_id() == model_regst_desc_id_) {
        CHECK_EQ(regst_wp->model_version_id(), expected_model_version_id_++);
      } else {
        // do nothing
      }
      num_of_read_empty_ -= read_regst_[regst_wp->regst_desc_id()].empty();
      read_regst_.at(regst_wp->regst_desc_id()).push(regst_wp);
    }
  }
  TryWardKernelAndSendMsg();
  return 0;
}

int BpDataCompActor::HandleBpCompWhenNoReadableRegstMsg(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()), 0);
  TryWardKernelAndSendMsg();
  if (read_regst_.at(activation_regst_desc_id_).empty()) {
    while (!read_regst_.at(model_regst_desc_id_).empty()) {
      AsyncSendRegstMsgToProducer(read_regst_.at(model_regst_desc_id_).front());
      read_regst_.at(model_regst_desc_id_).pop();
    }
    AsyncSendRegstMsgToProducer(read_regst_.at(model_tmp_regst_desc_id_).front());
    read_regst_.at(model_tmp_regst_desc_id_).pop();
    AsyncSendRegstDescDoneMsgForAllProducedRegstDesc();
    num_of_read_empty_ = 6;
    if (total_reading_cnt() == 0) {
      cur_msg_handle_ = nullptr;
      return 1;
    } else {
      cur_msg_handle_ = &BpDataCompActor::HandleWaitUntilReadingCntEqualZero;
      return 0;
    }
  }
  return 0;
}
  
int BpDataCompActor::HandleWaitUntilReadingCntEqualZero(
    const ActorMsg& msg,
    const ThreadContext& thread_ctx) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_warpper()->regst_raw_ptr()), 0);
  if (total_reading_cnt() == 0) {
    cur_msg_handle_ = nullptr;
    return 1;
  }
  return 0;
}

void BpDataCompActor::TryWardKernelAndSendMsg() {
  while (IsReadReady() && IsWriteReady()) {
    uint64_t cur_model = read_regst_.at(model_regst_desc_id_).front()->model_version_id();
    uint64_t piece_id = expected_piece_id();
    CHECK_EQ(cur_model, read_regst_.at(activation_regst_desc_id_).front()->model_version_id());
    CHECK_EQ(cur_model, read_regst_.at(data_tmp_regst_desc_id_).front()->model_version_id());
    for (const auto& pair : read_regst_) {
      if (pair.first != model_regst_desc_id_ && pair.first != model_tmp_regst_desc_id_) {
        CHECK_EQ(pair.second.front()->piece_id(), piece_id);
      }
    }
    AsyncWardKernel(GenDefaultKernelCtx(), 
        [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWarpper> {
      Regst* regst = GetCurWriteableRegst(regst_desc_id);
      if (regst == nullptr) {
        return read_regst_.at(regst_desc_id).front();
      } else {
        return std::make_shared<LocalRegstWarpper> (regst);
      }
    });
    ForEachCurWriteableRegst([piece_id](Regst* regst) {
      regst->set_piece_id(piece_id);
    });
    AsyncSendReadableRegstMsg();
    for (auto& pair : read_regst_) {
      if (pair.first != model_regst_desc_id_ && pair.first != model_tmp_regst_desc_id_) {
        AsyncSendRegstMsgToProducer(pair.second.front());
        pair.second.pop();
        num_of_read_empty_ += pair.second.empty();
      }
    }
  }
}

REGISTER_ACTOR(kDataCompTask, false, BpDataCompActor);

}  // namespace oneflow
