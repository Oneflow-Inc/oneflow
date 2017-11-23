#include "oneflow/core/actor/accumulate_actor.h"

namespace oneflow {

void AccumulateActor::Init(const TaskProto& task_proto, int32_t max_acc_cnt) {
  using namespace std::placeholders;
  is_in_eord_ = false;
  if (GetDeviceType() == DeviceType::kCPU) {
    cpy_func_ = std::bind(Memcpy<DeviceType::kCPU>, _1, _2, _3, _4,
                          cudaMemcpyKind::cudaMemcpyHostToHost);
  } else {
    cpy_func_ = std::bind(Memcpy<DeviceType::kGPU>, _1, _2, _3, _4,
                          cudaMemcpyKind::cudaMemcpyDeviceToDevice);
  }
  OF_SET_MSG_HANDLER(&AccumulateActor::HandlerNormal);
  acc_cnt_ = 0;
  max_acc_cnt_ = max_acc_cnt;
  next_piece_id_ = 0;
}

int AccumulateActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kEordMsg) {
    is_in_eord_ = true;
    DecreaseRemainingEordCnt();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      pending_in_regst_.push(regst);
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return TrySwitchToZombieOrFinish();
}

bool AccumulateActor::IsReadAlwaysUnReadyFromNow() {
  return is_in_eord_ && pending_in_regst_.empty();
}

void AccumulateActor::AsyncReturnAllReadableRegst() {
  CHECK(pending_in_regst_.empty());
}

void AccumulateActor::Act() {
  Regst* in_regst = pending_in_regst_.front();
  Regst* out_regst = GetCurSoleWriteableRegst();
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  if (acc_cnt_ == 0) {
    Blob* in_blob = in_regst->packed_blob();
    Blob* out_blob = out_regst->packed_blob();
    cpy_func_(kernel_ctx.device_ctx, out_blob->mut_dptr(), in_blob->dptr(),
              in_blob->ByteSizeOfDataField());
  } else {
    AsyncLaunchKernel(kernel_ctx, [this](uint64_t regst_desc_id) -> Regst* {
      Regst* regst = GetCurWriteableRegst(regst_desc_id);
      if (regst == nullptr) {
        CHECK_EQ(regst_desc_id, pending_in_regst_.front()->regst_desc_id());
        return pending_in_regst_.front();
      } else {
        return regst;
      }
    });
  }
  acc_cnt_ += 1;
  if (acc_cnt_ == max_acc_cnt_) {
    AsyncSendRegstMsgToConsumer(
        [&](Regst* regst) { regst->set_piece_id(next_piece_id_); });
    acc_cnt_ = 0;
    next_piece_id_ += 1;
  }
  AsyncSendRegstMsgToProducer(in_regst);
  pending_in_regst_.pop();
}

}  // namespace oneflow
