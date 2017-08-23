//#include "oneflow/core/kernel/accumulate_kernel.h"
//
// namespace oneflow {
//
// template<DeviceType device_type, typename FloatingPointType>
// void AccumulateKernel<device_type, FloatingPointType>::Forward(
//    const KernelCtx& ctx,
//    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
//  const Blob* in_blob = BnInOp2BlobPtr("one");
//  Blob* out_blob = BnInOp2BlobPtr("acc");
//  KernelUtil<device_type, FloatingPointType>::BlasAxpy(
//      ctx, in_blob->shape().elem_cnt(), static_cast<FloatingPointType>(1.0),
//      in_blob->dptr<FloatingPointType>(), 1,
//      out_blob->mut_dptr<FloatingPointType>(), 1);
//}
//
// INSTANTIATE_KERNEL_CLASS(AccumulateKernel);
// REGISTER_KERNEL(OperatorConf::kAccumulateConf, AccumulateKernel);
//
//}  // namespace oneflow
