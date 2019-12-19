#include "oneflow/core/actor/accumulate_compute_actor.h"

namespace oneflow {

void AccumulateCompActor::Init(const TaskProto& task_proto, int32_t max_acc_cnt, ColIdOrder order) {
  using namespace std::placeholders;
  order_ = order;
  if (GetDeviceType() == DeviceType::kCPU) {
    cpy_func_ = std::bind(Memcpy<DeviceType::kCPU>, _1, _2, _3, _4
#ifdef WITH_CUDA
                          ,
                          cudaMemcpyHostToHost
#endif
    );
  } else {
#ifdef WITH_CUDA
    cpy_func_ = std::bind(Memcpy<DeviceType::kGPU>, _1, _2, _3, _4, cudaMemcpyDeviceToDevice);
#else
    UNIMPLEMENTED();
#endif
  }
  OF_SET_MSG_HANDLER(&AccumulateCompActor::HandlerNormal);
  acc_cnt_ = 0;
  max_acc_cnt_ = max_acc_cnt;
  next_piece_id_ = 0;
}

int64_t AccumulateCompActor::ActNumForEachOutput(int64_t regst_desc_id) const {
  return regst_desc_id == Name2SoleRegstDescId("acc") ? max_acc_cnt_ : 1;
}

void AccumulateCompActor::Act() {
  Regst* in_regst = GetNaiveCurReadable("one");
  Regst* out_regst = GetNaiveCurWriteable("acc");
  KernelCtx kernel_ctx = GenDefaultKernelCtx();
  if (acc_cnt_ == 0 && IsFirstRegstInPieceWithOrder(in_regst, order_)) {
    Blob* in_blob = in_regst->packed_blob();
    Blob* out_blob = out_regst->packed_blob();
    cpy_func_(kernel_ctx.device_ctx, out_blob->mut_dptr(), in_blob->dptr(),
              in_blob->ByteSizeOfBlobBody());
  } else {
    AsyncLaunchKernel(kernel_ctx);
  }
  if (IsLastRegstInPieceWithOrder(in_regst, order_)) { acc_cnt_ += 1; }
}

void AccumulateCompActor::VirtualAsyncSendNaiveProducedRegstMsgToConsumer() {
  if (acc_cnt_ == max_acc_cnt_) {
    HandleProducedNaiveDataRegstToConsumer([&](Regst* regst) {
      regst->set_piece_id(next_piece_id_);
      return true;
    });
    acc_cnt_ = 0;
    next_piece_id_ += 1;
  }
}

}  // namespace oneflow
