//#include "oneflow/core/kernel/copy_hd_kernel.h"
//
// namespace oneflow {
//
// namespace {
//
// template<DeviceType device_type, typename FloatingPointType>
// void CopyH2DAsync(const KernelCtx& ctx, Blob* out_blob, Blob* in_blob,
//                  const size_t type_size) {
//  KernelUtil<device_type, FloatingPointType>::Memcpy(
//      ctx, out_blob->mut_dptr(), in_blob->dptr(),
//      in_blob->shape().elem_cnt() * type_size, cudaMemcpyHostToDevice);
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void CopyD2HAsync(const KernelCtx& ctx, Blob* out_blob, Blob* in_blob,
//                  const size_t type_size) {
//  KernelUtil<device_type, FloatingPointType>::Memcpy(
//      ctx, out_blob->mut_dptr(), in_blob->dptr(),
//      in_blob->shape().elem_cnt() * type_size, cudaMemcpyDeviceToHost);
//}
//
//}  // namespace
//
// template<DeviceType device_type, typename FloatingPointType>
// void CopyHdKernel<device_type, FloatingPointType>::InitFromOpProto(
//    const OperatorProto& op_proto) {
//  Kernel::InitFromOpProto(op_proto);
//
//  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();
//
//  if (copy_hd_conf.type() == CopyHdOpConf::H2D) {
//    ForwardCopyFunc_ = &CopyH2DAsync<device_type, FloatingPointType>;
//    BackwardCopyFunc_ = &CopyD2HAsync<device_type, FloatingPointType>;
//  } else {
//    ForwardCopyFunc_ = &CopyD2HAsync<device_type, FloatingPointType>;
//    BackwardCopyFunc_ = &CopyH2DAsync<device_type, FloatingPointType>;
//  }
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void CopyHdKernel<device_type, FloatingPointType>::Forward(
//    const KernelCtx& ctx,
//    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
//  Blob* in_blob = BnInOp2BlobPtr(op()->SoleIbn());
//  Blob* out_blob = BnInOp2BlobPtr(op()->SoleObn());
//
//  ForwardCopyFunc_(ctx, out_blob, in_blob, sizeof(FloatingPointType));
//}
//
// template<DeviceType device_type, typename FloatingPointType>
// void CopyHdKernel<device_type, FloatingPointType>::Backward(
//    const KernelCtx& ctx,
//    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
//  Blob* in_blob = BnInOp2BlobPtr(op()->SoleOdbn());
//  Blob* out_blob = BnInOp2BlobPtr(op()->SoleIdbn());
//
//  BackwardCopyFunc_(ctx, out_blob, in_blob, sizeof(FloatingPointType));
//}
//
// INSTANTIATE_GPU_KERNEL_CLASS(CopyHdKernel);
// REGISTER_GPU_KERNEL(OperatorConf::kCopyHdConf, CopyHdKernel);
//
//}  // namespace oneflow
