#include "oneflow/core/kernel/copy_hd_kernel.h"

namespace oneflow {

void CopyHdKernel::InitFromOpProto(const OperatorProto& op_proto) {
  Kernel::InitFromOpProto(op_proto);

  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();

  if (copy_hd_conf.type() == CopyHdOpConf::H2D) {
    fw_kind_ = cudaMemcpyKind::cudaMemcpyHostToDevice;
    bw_kind_ = cudaMemcpyKind::cudaMemcpyDeviceToHost;
  } else {
    fw_kind_ = cudaMemcpyKind::cudaMemcpyDeviceToHost;
    bw_kind_ = cudaMemcpyKind::cudaMemcpyHostToDevice;
  }
}

void CopyHdKernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(op()->SoleIbn());
  Blob* out_blob = BnInOp2Blob(op()->SoleObn());

  Memcpy<DeviceType::kGPU>(ctx.device_ctx, out_blob->mut_start_memory(),
                           in_blob->start_memory(), in_blob->TotalByteSize(),
                           fw_kind_);
}

void CopyHdKernel::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(op()->SoleOdbn());
  Blob* in_diff_blob = BnInOp2Blob(op()->SoleIdbn());

  Memcpy<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_start_memory(),
                           out_diff_blob->start_memory(),
                           out_diff_blob->TotalByteSize(), bw_kind_);
}

COMMAND(AddKernelCreator(OperatorConf::kCopyHdConf,
                         []() { return new CopyHdKernel; }));

}  // namespace oneflow
