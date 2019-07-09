#include "oneflow/core/kernel/cuda_copy_peer_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

void CudaCopyPeerKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob(op_attribute().input_bns(0));
  Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(0));
  out_blob->CopyValidDataContentFrom(ctx.device_ctx, in_blob);
}

REGISTER_KERNEL(OperatorConf::kCudaCopyPeerConf, CudaCopyPeerKernel);

#endif

}  // namespace oneflow
