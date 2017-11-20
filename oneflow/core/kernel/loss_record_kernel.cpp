#include "oneflow/core/kernel/loss_record_kernel.h"

namespace oneflow {

template<typename T>
void LossRecordKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* loss_acc_blob = BnInOp2Blob("loss_acc");
  T loss_mean = loss_acc_blob->dptr<T>()[0];
  loss_mean /= JobDesc::Singleton()->ParallelPieceSize()
               * JobDesc::Singleton()->PieceNumOfRecordLoss();
  LOG(INFO) << "loss: " << loss_mean;
}

}  // namespace oneflow
