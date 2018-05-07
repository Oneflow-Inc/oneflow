#include "oneflow/core/kernel/reduce_scatter_kernel.h"

namespace oneflow {

void ReduceScatterKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const char* src_cur_dptr = in_blob->dptr<char>();
  for (const std::string& obn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);
    size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
    AutoMemcpy(ctx.device_ctx, out_blob->mut_dptr<char>(), src_cur_dptr, out_byte_size,
               in_blob->mem_case(), out_blob->mem_case());
    src_cur_dptr += out_byte_size;
  }
}

REGISTER_KERNEL(OperatorConf::kReduceScatterConf, ReduceScatterKernel);

}  // namespace oneflow
