#include "oneflow/core/kernel/loss_print_kernel.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

template<typename T>
void LossPrintKernel<T>::Forward(const KernelCtx& kernel_ctx,
                                 std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (HasEmptyShapeBlob(this->op_attribute().input_bns(), BnInOp2Blob)) { return; }
  const Blob* loss_acc_blob = BnInOp2Blob("loss_acc");
  T loss_reduced = loss_acc_blob->dptr<T>()[0];
  T reduction_coefficient = -1.0;
  const Blob* reduction_acc_blob = BnInOp2Blob("reduction_acc");
  if (reduction_acc_blob != nullptr) {
    reduction_coefficient = reduction_acc_blob->dptr<T>()[0];
  } else {
    auto& conf = op_conf().loss_print_conf();
    reduction_coefficient = GetReductionCoefficient(conf.weight_scalar(), conf.reduction_type(),
                                                    BnInOp2Blob("loss_instance_num")->dptr<T>()[0]);
  }
  loss_reduced /= reduction_coefficient;
  bool tmp_split = this->job_desc().IsPredict()
                   && this->job_desc().other_conf().predict_conf().has_tmp_split_fw_bw_train_conf();
  const char* loss_op_name = op_conf().name().c_str() + (tmp_split ? 0 : LossPrintPrefix.length());
  double* prev_ts = static_cast<double*>(kernel_ctx.other);
  const double cur_ts = GetCurTime() / 1e9;
  if (*prev_ts == 0) {
    LOG(INFO) << loss_op_name << ":" << loss_reduced;
  } else {
    LOG(INFO) << loss_op_name << ":" << std::fixed << std::setprecision(3) << loss_reduced << " ("
              << (cur_ts - *prev_ts) << " sec)";
  }
  *prev_ts = cur_ts;
}

template<typename T>
T LossPrintKernel<T>::GetReductionCoefficient(T weight_scalar, ScalarReductionType type,
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
