#include "oneflow/core/actor/model_diff_accumulate_actor.h"

namespace oneflow {

void MdDiffAccActor::Init(const TaskProto& task_proto,
                          const ThreadCtx& thread_ctx) {
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
  set_num_of_not_eord(1);
  mut_num_of_read_empty() = 1;
  OF_SET_MSG_HANDLER(&MdDiffAccActor::HandlerNormal);
  ForEachCurWriteableRegst(
      [this](Regst* regst) { model_diff_acc_cnt_[regst] = 0; });
}

int MdDiffAccActor::HandlerNormal(const ActorMsg& msg) {
  if (msg.msg_type() == ActorMsgType::kCmdMsg) {
    CHECK_EQ(msg.actor_cmd(), ActorCmd::kEORD);
    ProcessEord();
  } else if (msg.msg_type() == ActorMsgType::kRegstMsg) {
    if (TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr())
        != 0) {
      mut_num_of_read_empty() = 0;
      waiting_in_regst_.push(msg.regst_wrapper());
    } else {
      // do nothing
    }
    ActUntilFail();
  }
  return msg_handler() == nullptr;
}

int MdDiffAccActor::HandlerWaitUntilNoReadableRegst(const ActorMsg& msg) {
  CHECK_EQ(TryUpdtStateAsProducedRegst(msg.regst_wrapper()->regst_raw_ptr()),
           0);
  ActUntilFail();
  if (waiting_in_regst_.empty()) {
    AsyncSendEORDMsgForAllProducedRegstDesc();
    OF_SET_MSG_HANDLER(&MdDiffAccActor::HandlerWaitUntilReadingCntEqualZero);
  }
  return 0;
}

void MdDiffAccActor::Act() {
  std::shared_ptr<RegstWrapper> regst_wp = waiting_in_regst_.front();
  CHECK_EQ(regst_wp->piece_id(), expected_piece_id());
  KernelCtx ctx = GenDefaultKernelCtx();
  ForEachCurWriteableRegst([&](Regst* regst) {
    auto diff_cnt = model_diff_acc_cnt_.find(regst);
    if (diff_cnt->second != JobDesc::Singleton()->num_of_pieces_in_batch()) {
      return;
    }
    Blob* packed_blob = regst->GetBlobPtrFromLbn(kPackedBlobName);
    MemsetFunc(ctx, packed_blob->mut_dptr(), 0,
               packed_blob->shape().elem_cnt()
                   * JobDesc::Singleton()->FloatingPointSize());
    diff_cnt->second = 0;
  });
  AsyncLaunchKernel(
      ctx, [this](uint64_t regst_desc_id) -> std::shared_ptr<RegstWrapper> {
        Regst* regst = GetCurWriteableRegst(regst_desc_id);
        if (regst == nullptr) {
          CHECK_EQ(regst_desc_id, waiting_in_regst_.front()->regst_desc_id());
          return waiting_in_regst_.front();
        } else {
          return std::make_shared<LocalRegstWrapper>(regst);
        }
      });
  AsyncSendReadableRegstMsg([this, &regst_wp](Regst* regst) {
    regst->set_piece_id(regst_wp->piece_id());
    ++model_diff_acc_cnt_.at(regst);
  });
  AsyncSendRegstMsgToProducer(regst_wp);
  waiting_in_regst_.pop();
  mut_num_of_read_empty() = waiting_in_regst_.empty();
}

}  // namespace oneflow
