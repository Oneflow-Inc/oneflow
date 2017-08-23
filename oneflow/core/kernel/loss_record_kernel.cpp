//#include "oneflow/core/kernel/loss_record_kernel.h"
//
// namespace oneflow {
//
// template<typename FloatingPointType>
// void LossRecordKernel<DeviceType::kCPU, FloatingPointType>::Forward(
//    const KernelCtx& kernel_ctx,
//    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
//  const Blob* loss_acc_blob = BnInOp2BlobPtr("loss_acc");
//  CHECK_EQ(loss_acc_blob->shape().elem_cnt(), 1);
//  FloatingPointType loss_mean = loss_acc_blob->dptr<FloatingPointType>()[0];
//  loss_mean /= JobDesc::Singleton()->piece_size()
//               * JobDesc::Singleton()->piece_num_of_record_loss();
//  LOG(INFO) << "loss: " << loss_mean;
//}
//
// INSTANTIATE_CPU_KERNEL_CLASS(LossRecordKernel);
// REGISTER_CPU_KERNEL(OperatorConf::kLossRecordConf, LossRecordKernel);
//
//}  // namespace oneflow
