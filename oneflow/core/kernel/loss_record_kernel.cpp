#include "oneflow/core/kernel/loss_record_kernel.h"

namespace oneflow {

template<typename T>
void LossRecordKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* loss_acc_blob = BnInOp2Blob("loss_acc");
  T loss_mean = loss_acc_blob->dptr<T>()[0];
  loss_mean /= JobDesc::Singleton()->piece_size()
               * JobDesc::Singleton()->piece_num_of_record_loss();
  LOG(INFO) << "loss: " << loss_mean;
}

namespace {

Kernel* CreateLossRecordKernel(const OperatorConf& op_conf) {
  static const HashMap<int, std::function<Kernel*()>> data_type2creator = {
#define LOSS_RECORD_KERNEL_ENTRY(type_cpp, type_proto) \
  {type_proto, []() { return new LossRecordKernel<type_cpp>; }},
      FOR_EACH_PAIR(LOSS_RECORD_KERNEL_ENTRY, FLOATING_DATA_TYPE_SEQ)};
  return data_type2creator.at(JobDesc::Singleton()->default_data_type())();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kLossRecordConf, DeviceType::kCPU,
                         CreateLossRecordKernel));

}  // namespace oneflow
