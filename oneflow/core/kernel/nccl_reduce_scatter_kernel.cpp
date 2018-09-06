#include "oneflow/core/kernel/nccl_reduce_scatter_kernel.h"
#include "oneflow/core/device/cuda_util.h"
#include "oneflow/core/device/nccl_util.h"

namespace oneflow {

void NcclReduceScatterKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* in_blob = BnInOp2Blob("in");
  Blob* out_blob = BnInOp2Blob("out");
  auto elem_cnt = (size_t)out_blob->shape().elem_cnt();
  NcclCheck(ncclReduceScatter(in_blob->dptr(), out_blob->mut_dptr(), elem_cnt,
                              GetNcclDataType(in_blob->data_type()), ncclSum,
                              ctx.device_ctx->nccl_handle(), ctx.device_ctx->cuda_stream()));
}

REGISTER_KERNEL(OperatorConf::kNcclReduceScatterConf, NcclReduceScatterKernel);

}  // namespace oneflow
