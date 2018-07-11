#include "oneflow/core/kernel/accuracy_print_kernel.h"

namespace oneflow {

template<typename T>
void AccuracyPrintKernel<T>::Forward(const KernelCtx& kernel_ctx,
                                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* accuracy_acc_blob = BnInOp2Blob("accuracy_acc");
  T accuracy_num = accuracy_acc_blob->dptr<T>()[0];
  int total_num = Global<JobDesc>::Get()->BatchSize();
  float accuracy = accuracy_num / total_num;
  const char* accuracy_op_name = op_conf().name().c_str() + 15;
  LOG(INFO) << accuracy_op_name << ":" << accuracy;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAccuracyPrintConf, AccuracyPrintKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow