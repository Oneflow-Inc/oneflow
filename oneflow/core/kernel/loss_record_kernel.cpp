#include "oneflow/core/kernel/loss_record_kernel.h"

namespace oneflow {

template<typename T>
void LossRecordKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* loss_acc_blob = BnInOp2Blob("loss_acc");
  T loss_mean = loss_acc_blob->dptr<T>()[0];
  loss_mean /= JobDesc::Singleton()->piece_size()
               * JobDesc::Singleton()->TotalMachineNum()
               * JobDesc::Singleton()->piece_num_of_record_loss();
  LOG(INFO) << "loss: " << loss_mean;
}

Kernel* CreateLossRecordKernel() {
  static const HashMap<int, std::function<Kernel*()>> creators = {
#define LOSS_RECORD_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, []() { return new LossRecordKernel<type_cpp>; }},
      OF_PP_FOR_EACH_TUPLE(LOSS_RECORD_KERNEL_ENTRY, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(JobDesc::Singleton()->default_data_type())();
}

COMMAND(AddKernelCreator(OperatorConf::kLossRecordConf,
                         CreateLossRecordKernel));

}  // namespace oneflow
