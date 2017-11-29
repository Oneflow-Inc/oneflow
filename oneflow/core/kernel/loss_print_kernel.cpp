#include "oneflow/core/kernel/loss_print_kernel.h"

namespace oneflow {

template<typename T>
void LossPrintKernel<T>::Forward(
    const KernelCtx& kernel_ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* loss_acc_blob = BnInOp2Blob("loss_acc");
  T loss_mean = loss_acc_blob->dptr<T>()[0];
  loss_mean /= JobDesc::Singleton()->ParallelPieceSize()
               * JobDesc::Singleton()->PieceNumOfPrintLoss();
  LOG(INFO) << "loss: " << loss_mean;
}

namespace {

Kernel* CreateLossPrintKernel(const KernelConf& kernel_conf) {
  static const HashMap<int, std::function<Kernel*()>> creators = {
#define MAKE_ENTRY(type_cpp, type_proto) \
  {type_proto, []() { return new LossPrintKernel<type_cpp>; }},
      OF_PP_FOR_EACH_TUPLE(MAKE_ENTRY, FLOATING_DATA_TYPE_SEQ)};
  return creators.at(kernel_conf.data_type())();
}

}  // namespace

COMMAND(AddKernelCreator(OperatorConf::kLossPrintConf, CreateLossPrintKernel));

}  // namespace oneflow
