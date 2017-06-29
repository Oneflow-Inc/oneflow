#include "oneflow/core/kernel/copy_hd_kernel.h"

namespace oneflow {

namespace {

template<DeviceType device_type, typename floating_point_type>
void CopyH2DAsync(const KernelCtx& ctx, Blob* out_blob, Blob* in_blob,
                  const size_t type_size) {
  KernelUtil<device_type, floating_point_type>::Memcpy(
      ctx, out_blob->mut_dptr(), in_blob->dptr(),
      in_blob->shape().elem_cnt() * type_size, cudaMemcpyHostToDevice);
}

template<DeviceType device_type, typename floating_point_type>
void CopyD2HAsync(const KernelCtx& ctx, Blob* out_blob, Blob* in_blob,
                  const size_t type_size) {
  KernelUtil<device_type, floating_point_type>::Memcpy(
      ctx, out_blob->mut_dptr(), in_blob->dptr(),
      in_blob->shape().elem_cnt() * type_size, cudaMemcpyDeviceToHost);
}

}  // namespace

template<DeviceType device_type, typename floating_point_type>
void CopyHdKernel<device_type, floating_point_type>::InitFromOpProto(
    const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);

  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();

  if (copy_hd_conf.type() == CopyHdOpConf::H2D) {
    ForwardCopyFunc = CopyH2DAsync;
    BackwardCopyFunc = CopyD2HAsync;
  } else {
    ForwardCopyFunc = CopyD2HAsync;
    BackwardCopyFunc = CopyH2DAsync;
  }
}

template<DeviceType device_type, typename floating_point_type>
void CopyHdKernel<device_type, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_blob  = BnInOp2BlobPtr(op()->SoleIbn());
  Blob* out_blob = BnInOp2BlobPtr(op()->SoleObn());

  (*ForwardCopyFunc)<device_type, floating_point_type>(
      ctx, out_blob, in_blob, sizeof(floating_point_type));
}

template<DeviceType device_type, typename floating_point_type>
void CopyHdKernel<device_type, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* in_blob  = BnInOp2BlobPtr(op()->SoleOdbn());
  Blob* out_blob = BnInOp2BlobPtr(op()->SoleIdbn());

  (*BackwardCopyFunc)<device_type, floating_point_type>(
      ctx, out_blob, in_blob, sizeof(floating_point_type));
}

INSTANTIATE_KERNEL_CLASS(CopyHdKernel);
REGISTER_KERNEL(OperatorConf::kCopyHdConf, CopyHdKernel);

}  // namespace oneflow
