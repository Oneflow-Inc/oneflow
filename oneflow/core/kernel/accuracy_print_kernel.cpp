#include "oneflow/core/kernel/accuracy_print_kernel.h"
#include "oneflow/core/job/keyword.h"

namespace oneflow {

template<typename T>
void AccuracyPrintKernel<T>::Forward(const KernelCtx& kernel_ctx,
                                     std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  if (HasEmptyShapeBlob(this->op_attribute().input_bns(), BnInOp2Blob)) { return; }
  const Blob* accuracy_acc_blob = BnInOp2Blob("accuracy_acc");
  const Blob* accuracy_instance_num_blob = BnInOp2Blob("accuracy_instance_num");
  T accurate_num = accuracy_acc_blob->dptr<T>()[0];
  int32_t accuracy_instance_num = static_cast<int32_t>(accuracy_instance_num_blob->dptr<T>()[0]);
  float accuracy = accurate_num / accuracy_instance_num;
  const char* accuracy_op_name = op_conf().name().c_str() + AccuracyPrintPrefix.length();
  auto kernel_conf = this->kernel_conf();
  const int32_t top_k_print =
      kernel_conf.op_attribute().op_conf().accuracy_print_conf().top_k_print();
  LOG(INFO) << "top_" << top_k_print << "_" << accuracy_op_name << ":" << accuracy;
}

ADD_CPU_DEFAULT_KERNEL_CREATOR(OperatorConf::kAccuracyPrintConf, AccuracyPrintKernel,
                               FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
