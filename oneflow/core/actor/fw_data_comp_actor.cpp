#include "oneflow/core/actor/fw_data_comp_actor.h"
#include "oneflow/core/actor/actor_registry.h"
#include "oneflow/core/register/local_register_wrapper.h"

namespace oneflow {

void FwDataCompActor::Init(const TaskProto& task_proto,
                           const ThreadCtx& thread_ctx) {
  CompActor::Init(task_proto, thread_ctx);
  in_desc_id_ = RegstDescId4Name("in");
  model_regst_desc_id_ = RegstDescId4Name("model");
  model_tmp_regst_desc_id_ = RegstDescId4Name("model_tmp");
  expected_model_version_id_ = 0;
  if (thread_ctx.cpu_stream) {
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  } else {
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  kernel_ctx_ = GenDefaultKernelCtx();
  if (in_desc_id_ == -1) {
    CHECK_EQ(model_regst_desc_id_, -1);
    CHECK_EQ(model_tmp_regst_desc_id_, -1);
    kernel_ctx_.other = reinterpret_cast<void*>(parallel_id());
    OF_SET_MSG_HANDLE(&FwDataCompActor::WaitToStart);
  } else {
    mut_num_of_not_eord() =
        1 + (model_regst_desc_id_ != -1) + (model_tmp_regst_desc_id_ != -1);
    mut_num_of_read_empty() = 1;  // only consider "in"regst
    OF_SET_MSG_HANDLE(&FwDataCompActor::HandleNormal);
  }
  bp_actor_id_ = IDMgr::Singleton()->ActorId4TaskId(task_proto.bp_task_id());
}

bool FwDataCompActor::IsReadReady() {
  if (in_desc_id_ == -1) { return true; }
  if (in_.empty() || (model_regst_desc_id_ != -1 && !model_regst_)
      || (model_tmp_regst_desc_id_ != -1 && !model_tmp_regst_)) {
    return false;
  }
  if (model_regst_desc_id_ != -1) {
    // Ho Q, Cipar J, Cui H, et al. More effective distributed ml via a stale
    // synchronous parallel parameter server
    int32_t staleness = JobDesc::Singleton()->staleness();
    int32_t num_of_piece_in_batch =
        JobDesc::Singleton()->num_of_piece_in_batch();
    int64_t cur_iteration = in_.front()->piece_id() / num_of_piece_in_batch;
    int64_t stale_version = cur_iteration - staleness;
    return model_regst_->model_version_id() >= stale_version;
  }
  return true;
}

int FwDataCompActor::WaitToStart(const ActorMsg& msg) {
  CHECK_EQ(msg.actor_cmd(), ActorCmd::kStart);
  ActUntilFail();
  OF_SET_MSG_HANDLE(&FwDataCompActor::HandleWaitUntilNoReadableRegst);
  return 0;
}

void FwDataCompActor::AsyncSendMsgToModelAndModelTmpProducer() {
  if (model_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(model_regst_);
    model_regst_ = nullptr;
  }
  if (model_tmp_regst_desc_id_ != -1) {
    AsyncSendRegstMsgToProducer(model_tmp_regst_);
    model_tmp_regst_ = nullptr;
  }
}

int FwDataCompActor::HandleNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
    if (msg_handle() == &FwDataCompActor::HandleWaitUntilReadingCntEqualZero
        || msg_handle() == nullptr) {
      AsyncSendMsgToModelAndModelTmpProducer();
    }
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr())
        != 0) {
      std::shared_ptr<RegstWrapper> regst_wp = msg.regst_wrapper();
      if (regst_wp->regst_desc_id() == model_tmp_regst_desc_id_) {
        CHECK(!model_tmp_regst_);
        model_tmp_regst_ = regst_wp;
        readable_regst_[model_tmp_regst_desc_id_] = regst_wp;
      } else if (regst_wp->regst_desc_id() == model_regst_desc_id_) {
        CHECK_EQ(regst_wp->model_version_id(), expected_model_version_id_);
        if (model_regst_) { AsyncSendRegstMsgToProducer(model_regst_); }
        model_regst_ = regst_wp;
        readable_regst_[model_regst_desc_id_] = regst_wp;
        expected_model_version_id_ += 1;
      } else {
        mut_num_of_read_empty() -= in_.empty();
        in_.push(regst_wp);
      }
    }
  }
  ActUntilFail();
  return 0;
}

int FwDataCompActor::HandleWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr()),
           0);
  ActUntilFail();
  int total_piece_num = JobDesc::Singleton()->total_piece_num();
  if ((in_desc_id_ != -1 && in_.empty())
      || expected_piece_id() == total_piece_num) {
    AsyncSendMsgToModelAndModelTmpProducer();
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLE(&FwDataCompActor::HandleWaitUntilReadingCntEqualZero);
  }
  return 0;
}

void FwDataCompActor::Act() {
  int64_t piece_id = expected_piece_id();
  if (!in_.empty()) {
    CHECK_EQ(in_.front()->piece_id(), piece_id);
    readable_regst_[in_.front()->regst_desc_id()] = in_.front();
  }
  int64_t model_version_id = -1;
  if (model_regst_) { model_version_id = model_regst_->model_version_id(); }
  AsyncLaunchKernel(
      kernel_ctx_,
      [this](int64_t regst_desc_id) -> std::shared_ptr<RegstWrapper> {
        Regst* regst = GetCurWriteableRegst(regst_desc_id);
        if (regst == nullptr) {
          return readable_regst_.at(regst_desc_id);
        } else {
          return std::make_shared<LocalRegstWrapper>(regst);
        }
      });
  AsyncSendReadableRegstMsg([piece_id, model_version_id](Regst* regst) {
    regst->set_piece_id(piece_id);
    regst->set_model_version_id(model_version_id);
  });
  if (!in_.empty()) {
    AsyncSendRegstMsgToProducer(in_.front());
    in_.pop();
    mut_num_of_read_empty() += in_.empty();
  }
  if (bp_actor_id_ != -1) {
    ActorMsg msg;
    msg.set_dst_actor_id(bp_actor_id_);
    msg.set_piece_id(piece_id);
    msg.set_model_version_id(model_version_id);
    AsyncDo([msg]() { ActorMsgBus::Singleton()->SendMsg(msg); });
  }
}

REGISTER_ACTOR(kDataCompTask, true, FwDataCompActor);

}  // namespace oneflow
