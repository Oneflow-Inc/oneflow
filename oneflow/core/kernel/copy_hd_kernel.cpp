#include "oneflow/core/kernel/copy_hd_kernel.h"

namespace oneflow {

#ifdef WITH_CUDA

void CopyHdKernel::ForwardDataContent(const KernelCtx& ctx,
                                      std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  auto kernel_conf = this->kernel_conf();
  CHECK(kernel_conf.has_copy_hd_conf());
  bool enable_synthetic_data = kernel_conf.copy_hd_conf().enable_synthetic_data();
  const Blob* in_blob = BnInOp2Blob(op_attribute().input_bns(0));
  Blob* out_blob = BnInOp2Blob(op_attribute().output_bns(0));
  if (!enable_synthetic_data || act_cnt_ < 2) {
    out_blob->CopyDataContentFrom(ctx.device_ctx, in_blob);
    ++act_cnt_;
  }
}

REGISTER_KERNEL(OperatorConf::kCopyHdConf, CopyHdKernel);

#endif

}  // namespace oneflow
