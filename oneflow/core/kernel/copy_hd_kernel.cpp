#include "oneflow/core/kernel/copy_hd_kernel.h"

namespace oneflow {

// void CopyHdKernel::InitFromOpProto(const OperatorProto& op_proto) {
//  Kernel::InitFromOpProto(op_proto);
//
//  const CopyHdOpConf& copy_hd_conf = op()->op_conf().copy_hd_conf();
//
//  if (copy_hd_conf.type() == CopyHdOpConf::H2D) {
//    fw_kind_ = cudaMemcpyKind::cudaMemcpyHostToDevice;
//    bw_kind_ = cudaMemcpyKind::cudaMemcpyDeviceToHost;
//  } else {
//    fw_kind_ = cudaMemcpyKind::cudaMemcpyDeviceToHost;
//    bw_kind_ = cudaMemcpyKind::cudaMemcpyHostToDevice;
//  }
//}

void CopyHdKernel::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(kernel_conf().input_bns(0));
  Blob* out_blob = BnInOp2Blob(kernel_conf().output_bns(0));

  Memcpy<DeviceType::kGPU>(ctx.device_ctx, out_blob->mut_memory_ptr(),
                           in_blob->memory_ptr(), in_blob->TotalByteSize(),
                           fw_kind_);
}

void CopyHdKernel::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* out_diff_blob = BnInOp2Blob(kernel_conf().output_diff_bns(0));
  Blob* in_diff_blob = BnInOp2Blob(kernel_conf().input_diff_bns(0));

  Memcpy<DeviceType::kGPU>(ctx.device_ctx, in_diff_blob->mut_memory_ptr(),
                           out_diff_blob->memory_ptr(),
                           out_diff_blob->TotalByteSize(), bw_kind_);
}

}  // namespace oneflow
