#include "oneflow/core/kernel/reduce_gather_kernel.h"

namespace oneflow {

void ReduceGatherKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  Blob* out_blob = BnInOp2Blob("out");
  char* dst_cur_dptr = out_blob->mut_dptr<char>();
  for (const std::string& ibn : this->op_attribute().input_bns()) {
    Blob* in_blob = BnInOp2Blob(ibn);
    if (!in_blob) { break; }
    size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
    AutoMemcpy(ctx.device_ctx, dst_cur_dptr, in_blob->dptr<char>(), in_byte_size,
               in_blob->mem_case(), out_blob->mem_case());
    dst_cur_dptr += in_byte_size;
  }
  CHECK_EQ(dst_cur_dptr - out_blob->mut_dptr<char>(), out_blob->ByteSizeOfDataContentField());
}

REGISTER_KERNEL(OperatorConf::kReduceGatherConf, ReduceGatherKernel);

}  // namespace oneflow
