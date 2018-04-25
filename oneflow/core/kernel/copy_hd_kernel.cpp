#include "oneflow/core/kernel/copy_hd_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

void CopyHdKernel::VirtualKernelInit(const ParallelContext*) {
  const CopyHdOpConf& copy_hd_conf = op_conf().copy_hd_conf();

  if (copy_hd_conf.type() == CopyHdOpConf::H2D) {
    cp_kind_ = cudaMemcpyKind::cudaMemcpyHostToDevice;
  } else {
    cp_kind_ = cudaMemcpyKind::cudaMemcpyDeviceToHost;
  }
}

void CopyHdKernel::Forward(const KernelCtx& ctx,
                           std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(op_attribute().input_bns(0));
  Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(0));

  Memcpy<DeviceType::kGPU>(ctx.device_ctx, out_blob->mut_memory_ptr(), in_blob->memory_ptr(),
                           in_blob->TotalByteSize(), cp_kind_);
}

REGISTER_KERNEL(OperatorConf::kCopyHdConf, CopyHdKernel);

#endif

}  // namespace oneflow
