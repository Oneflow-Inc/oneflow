//#include "oneflow/core/kernel/clone_kernel.h"
//
// namespace oneflow {
//
// template<DeviceType device_type, typename FloatingPointType>
// void CloneKernel<device_type, FloatingPointType>::Forward(
//    const KernelCtx& ctx,
//    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
//  const Blob* in_blob = BnInOp2BlobPtr(op()->SoleIbn());
//  for (const std::string& obn : op()->output_bns()) {
//    Blob* out_blob = BnInOp2BlobPtr(obn);
//    KernelUtil<device_type, FloatingPointType>::Memcpy(
//        ctx, out_blob->mut_dptr(), in_blob->dptr(),
//        in_blob->shape().elem_cnt() * sizeof(FloatingPointType));
//  }
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void CloneKernel<device_type, FloatingPointType>::Backward(
//    const KernelCtx& ctx,
//    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
//  Blob* idbn_blob = BnInOp2BlobPtr(op()->SoleIdbn());
//  const std::vector<std::string>& odbns = op()->output_diff_bns();
//  if (odbns.size() == 0) return;
//  const Blob* odbn_blob_0 = BnInOp2BlobPtr(odbns[0]);
//  KernelUtil<device_type, FloatingPointType>::Memcpy(
//      ctx, idbn_blob->mut_dptr(), odbn_blob_0->dptr(),
//      idbn_blob->shape().elem_cnt() * sizeof(FloatingPointType));
//  for (size_t i = 1; i != odbns.size(); ++i) {
//    const Blob* odbn_blob = BnInOp2BlobPtr(odbns[i]);
//    KernelUtil<device_type, FloatingPointType>::BlasAxpy(
//        ctx, idbn_blob->shape().elem_cnt(), 1.0,
//        odbn_blob->dptr<FloatingPointType>(), 1,
//        idbn_blob->mut_dptr<FloatingPointType>(), 1);
//  }
//}
//
// INSTANTIATE_KERNEL_CLASS(CloneKernel);
// REGISTER_KERNEL(OperatorConf::kCloneConf, CloneKernel);
//
//}  // namespace oneflow
