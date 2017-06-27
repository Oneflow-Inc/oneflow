#include "oneflow/core/kernel/clone_kernel.h"
#include <string>
#include "oneflow/core/blas/kernel_util.h"

namespace oneflow {

template<DeviceType device_type, typename floating_point_type>
void CloneKernel<device_type, floating_point_type>::Forward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  const Blob* in_blob = BnInOp2BlobPtr(op()->SoleIbn());
  for(const std::string& obn : op()->output_bns()) {
    Blob* out_blob = BnInOp2BlobPtr(obn);
    ctx.device_ctx->cpu_stream()->Send([=] {
        KernelUtil<device_type, floating_point_type>::Memcpy(ctx, out_blob->mut_dptr(),
             in_blob->dptr(),
             in_blob->shape().elem_cnt() * sizeof(floating_point_type));
    });
  }
}

template<DeviceType device_type, typename floating_point_type>
void CloneKernel<device_type, floating_point_type>::Backward(
    const KernelCtx& ctx,
    std::function<Blob*(const std::string&)> BnInOp2BlobPtr) const {
  Blob* idbn_blob = BnInOp2BlobPtr(op()->SoleIdbn());
  const std::vector<std::string>& odbns = op()->output_diff_bns();
  if (odbns.size() == 0) return;
  const Blob* odbn_blob_0 = BnInOp2BlobPtr(odbns[0]);
  ctx.device_ctx->cpu_stream()->Send([=] {
      KernelUtil<device_type, floating_point_type>::Memcpy(ctx, idbn_blob->mut_dptr(),
           odbn_blob_0->dptr(),
           idbn_blob->shape().elem_cnt() * sizeof(floating_point_type));
  });
  for(size_t i = 1; i != odbns.size(); ++i) {
    const Blob* odbn_blob = BnInOp2BlobPtr(odbns[i]);
    ctx.device_ctx->cpu_stream()->Send([=] {
      KernelUtil<device_type, floating_point_type>::BlasAxpy( ctx,
          idbn_blob->shape().elem_cnt(), 1.0,
          static_cast<const floating_point_type*>(odbn_blob->dptr()), 1,
          static_cast<floating_point_type*>(idbn_blob->mut_dptr()), 1);
    });
  }
}

INSTANTIATE_CPU_KERNEL_CLASS(CloneKernel);
REGISTER_CPU_KERNEL(OperatorConf::kCloneConf, CloneKernel);

}  // namespace oneflow
