#include "oneflow/core/actor/accumulate_actor.h"
#include "oneflow/core/actor/actor_registry.h"

namespace oneflow {

void AccumulateActor::Init(const TaskProto& task_proto,
                           const ThreadCtx& thread_ctx, int32_t max_acc_cnt) {
  CompActor::Init(task_proto, thread_ctx);
  if (thread_ctx.cpu_stream) {
    MemsetFunc = &KernelUtil<DeviceType::kCPU, float>::Memset;
    mut_device_ctx().reset(new CpuDeviceCtx(thread_ctx.cpu_stream));
  } else {
    MemsetFunc = &KernelUtil<DeviceType::kGPU, float>::Memset;
    mut_device_ctx().reset(new CudaDeviceCtx(cuda_handle_.cuda_stream(),
                                             cuda_handle_.cublas_handle(),
                                             cuda_handle_.cudnn_handle()));
  }
  set_num_of_remaining_eord(1);
  mut_num_of_read_empty() = 1;
  OF_SET_MSG_HANDLER(&AccumulateActor::HandlerNormal);
  acc_cnt_ = max_acc_cnt;
  max_acc_cnt_ = max_acc_cnt;
  next_acc_piece_id_ = 0;
}

int AccumulateActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    Regst* regst = msg.regst();
    if (TryUpdtStateAsProducedRegst(regst) != 0) {
      mut_num_of_read_empty() = 0;
      waiting_in_regst_.push(regst);
      VLOG(4) << "accumulate actor " << actor_id() << " "
              << "receive readable regst " << regst << ", "
              << "regst_desc_id:" << regst->regst_desc_id() << ", "
              << "current num_of_read_empty:" << num_of_read_empty();
    }
    ActUntilFail();
  } else {
    UNEXPECTED_RUN();
  }
  return msg_handler() == nullptr;
}

int AccumulateActor::HandlerWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst()), 0);
  ActUntilFail();
  if (waiting_in_regst_.empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&AccumulateActor::HandlerZombie);
  }
  return 0;
}

void AccumulateActor::Act() {
  Regst* in_regst = waiting_in_regst_.front();
  CHECK_EQ(in_regst->piece_id(), expected_piece_id());
  KernelCtx ctx = GenDefaultKernelCtx();
  ForEachCurWriteableRegst([&](Regst* regst) {
    if (acc_cnt_ != max_acc_cnt_) { return; }
    Blob* packed_blob = regst->GetBlobPtrFromLbn(kPackedBlobName);
    MemsetFunc(ctx, packed_blob->mut_dptr(), 0, packed_blob->TotalByteSize());
    acc_cnt_ = 0;
  });
  AsyncLaunchKernel(ctx, [this](uint64_t regst_desc_id) -> Regst* {
    Regst* regst = GetCurWriteableRegst(regst_desc_id);
    if (regst == nullptr) {
      CHECK_EQ(regst_desc_id, waiting_in_regst_.front()->regst_desc_id());
      return waiting_in_regst_.front();
    } else {
      return regst;
    }
  });
  acc_cnt_ += 1;
  if (acc_cnt_ == max_acc_cnt_) {
    AsyncSendReadableRegstMsg([&](Regst* acc_regst) {
      acc_regst->set_piece_id(next_acc_piece_id_++);
    });
  }
  AsyncSendRegstMsgToProducer(in_regst);
  waiting_in_regst_.pop();
  mut_num_of_read_empty() = waiting_in_regst_.empty();
}

}  // namespace oneflow
