#include "oneflow/core/kernel/loss_print_kernel.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

template<typename T>
void LossPrintKernel<T>::Forward(const KernelCtx& kernel_ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (HasEmptyShapeBlob(this->op_attribute().input_bns(), BnInOp2Blob)) { return; }
  const Blob* loss_acc_blob = BnInOp2Blob("loss_acc");
  const Blob* total_instance_num_blob = BnInOp2Blob("batch_instance_num");
  const Blob* reduction_acc_blob = BnInOp2Blob("reduction_acc");
  T loss_reduced = loss_acc_blob->dptr<T>()[0];
  T reduction_coefficient = -1.0;
  if (total_instance_num_blob != nullptr) {
    reduction_coefficient = total_instance_num_blob->dptr<T>()[0];
  } else if (reduction_acc_blob != nullptr) {
    reduction_coefficient = reduction_acc_blob->dptr<T>()[0];
  } else {
    auto conf = op_conf().loss_print_conf();
    reduction_coefficient = GetReductionCoefficient(
        conf.weight_scalar(), conf.reduction_type(),
        Global<JobDesc>::Get()->PieceSize() * Global<JobDesc>::Get()->PieceNumOfPrintLoss());
  }
  loss_reduced /= reduction_coefficient;
  const char* loss_op_name = op_conf().name().c_str() + LossPrintPrefix.length();
  LOG(INFO) << loss_op_name << ":" << loss_reduced;
}

template<typename T>
T LossPrintKernel<T>::GetReductionCoefficient(T weight_scalar, LossReductionType type,
                                              int32_t n) const {
  switch (type) {
    case kSumOverOne: return 1.0;
    case kSumOverN: return n * 1.0;
    case kSumOverWeight: return n * weight_scalar;
    case kSumOverNonZeroWeight: return n * 1.0;
    default: UNIMPLEMENTED(); return -1.0;
  }
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kLossPrintConf, LossPrintKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
