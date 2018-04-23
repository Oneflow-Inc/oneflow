#include "oneflow/core/kernel/copy_hd_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

void CopyHdKernel::VirtualKernelInit(const ParallelContext*) {
  const CopyHdOpConf& copy_hd_conf = kernel_conf().op_conf().copy_hd_conf();

  if (copy_hd_conf.type() == CopyHdOpConf::H2D) {
    cp_kind_ = cudaMemcpyKind::cudaMemcpyHostToDevice;
  } else {
    cp_kind_ = cudaMemcpyKind::cudaMemcpyDeviceToHost;
  }
}

void CopyHdKernel::ForwardDataContent(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(kernel_conf().input_bns(0));
  Blob* out_blob = BnInOp2Blob(kernel_conf().output_bns(0));

  Memcpy<DeviceType::kGPU>(ctx.device_ctx, out_blob->mut_dptr(),
                           in_blob->dptr(),
                           in_blob->ByteSizeOfDataContentField(), cp_kind_);
}

COMMAND(AddKernelCreator(OperatorConf::kCopyHdConf,
                         []() { return new CopyHdKernel; }));

#endif

}  // namespace oneflow
