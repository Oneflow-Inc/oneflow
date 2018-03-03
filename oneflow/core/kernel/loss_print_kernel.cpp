#include "oneflow/core/kernel/loss_print_kernel.h"

namespace oneflow {

template<typename T>
void LossPrintKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* loss_acc_blob = BnInOp2Blob("loss_acc");
  const Blob* reduction_acc_blob = BnInOp2Blob("reduction_acc");
  T loss_mean = loss_acc_blob->dptr<T>()[0];
  T reduction_coefficient = reduction_acc_blob->dptr<T>()[0];
  loss_mean /= reduction_coefficient;
  const char* loss_op_name = op_conf().name().c_str() + 11;
  LOG(INFO) << loss_op_name << ":" << loss_mean;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLossPrintConf, LossPrintKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
