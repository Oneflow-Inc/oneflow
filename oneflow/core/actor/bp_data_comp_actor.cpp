#include "oneflow/core/actor/bp_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_wrapper.h"

namespace oneflow {

void BpDataCompActor::Init(const TaskProto& task_proto,
                           const ThreadCtx& thread_ctx) {
  Actor::Init(task_proto, thread_ctx);
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  activation_regst_desc_id_ = RegstDescId4Name("activation");
  data_tmp_regst_desc_id_ = RegstDescId4Name("data_tmp");
  out_regst_desc_id_ = RegstDescId4Name("out");
  expected_model_version_id_ = 0;
  mut_num_of_read_empty() =
      3 + (model_regst_desc_id_ != -1) + (model_tmp_regst_desc_id_ != -1)
      + (activation_regst_desc_id_ != -1) + (data_tmp_regst_desc_id_ != -1);
  set_num_of_not_eord(mut_num_of_read_empty());
  if (thread_ctx.cpu_stream) {
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  OF_SET_MSG_HANDLER(&BpDataCompActor::HandlerNormal);
}

bool BpDataCompActor::IsReadReady() {
  if (mut_num_of_read_empty()) { return false; }
  if (model_regst_desc_id_ != -1) {
    int cur_model_version_id =
        read_regst_.at(out_regst_desc_id_).front()->model_version_id();
    CHECK_GE(cur_model_version_id, 0);
    while (read_regst_.at(model_regst_desc_id_).front()->model_version_id()
               != cur_model_version_id
           && !read_regst_.at(model_regst_desc_id_).empty()) {
      AsyncSendRegstMsgToProducer(read_regst_.at(model_regst_desc_id_).front());
      read_regst_.at(model_regst_desc_id_).pop();
    }
    mut_num_of_read_empty() += read_regst_.at(model_regst_desc_id_).empty();
  }
  return !mut_num_of_read_empty();
}

void BpDataCompActor::AsyncSendMsgToModelAndModelTmpProducer() {
  while (model_regst_desc_id_ != -1
         && !read_regst_.at(model_regst_desc_id_).empty()) {
    AsyncSendRegstMsgToProducer(read_regst_.at(model_regst_desc_id_).front());
    read_regst_.at(model_regst_desc_id_).pop();
  }
  if (model_tmp_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(
        read_regst_.at(model_tmp_regst_desc_id_).front());
    read_regst_.at(model_tmp_regst_desc_id_).pop();
  }
}

int BpDataCompActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
    if (msg_handler() == &BpDataCompActor::HandlerWaitUntilReadingCntEqualZero
        || msg_handler() == nullptr) {
      AsyncSendMsgToModelAndModelTmpProducer();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr())
        != 0) {
      std::shared_ptr<RegstWrapper> regst_wp = msg.regst_wrapper();
      if (regst_wp->regst_desc_id() == model_tmp_regst_desc_id_) {
        CHECK(read_regst_.find(model_tmp_regst_desc_id_) == read_regst_.end());
      } else if (regst_wp->regst_desc_id() == model_regst_desc_id_) {
        CHECK_EQ(regst_wp->model_version_id(), expected_model_version_id_++);
      } else {
        // do nothing
      }
      mut_num_of_read_empty() -= read_regst_[regst_wp->regst_desc_id()].empty();
      read_regst_.at(regst_wp->regst_desc_id()).push(regst_wp);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int BpDataCompActor::HandlerWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr()),
           0);
  ActUntilFail();
  if (mut_num_of_read_empty()) {
    AsyncSendMsgToModelAndModelTmpProducer();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&BpDataCompActor::HandlerWaitUntilReadingCntEqualZero);
  }
  return 0;
}

void BpDataCompActor::Act() {
  int64_t piece_id = expected_piece_id();
  CHECK_EQ(read_regst_.at(out_regst_desc_id_).front()->piece_id(), piece_id);
  for (const auto& pair : read_regst_) {
    if (pair.first != model_regst_desc_id_
        && pair.first != model_tmp_regst_desc_id_) {
      CHECK_EQ(pair.second.front()->piece_id(), piece_id);
    }
  }
  AsyncLaunchKernel(
      GenDefaultKernelCtx(),
      [this](int64_t regst_desc_id) -> std::shared_ptr<RegstWrapper> {
        Regst* regst = GetCurWriteableRegst(regst_desc_id);
        if (regst == nullptr) {
          return read_regst_.at(regst_desc_id).front();
        } else {
          return std::make_shared<LocalRegstWrapper>(regst);
        }
      });
  AsyncSendReadableRegstMsg(
      [piece_id](Regst* regst) { regst->set_piece_id(piece_id); });
  for (auto& pair : read_regst_) {
    if (pair.first != model_regst_desc_id_
        && pair.first != model_tmp_regst_desc_id_) {
      AsyncSendRegstMsgToProducer(pair.second.front());
      pair.second.pop();
      mut_num_of_read_empty() += pair.second.empty();
    }
  }
}

REGISTER_ACTOR(kDataCompTask, false, BpDataCompActor);

}  // namespace oneflow
