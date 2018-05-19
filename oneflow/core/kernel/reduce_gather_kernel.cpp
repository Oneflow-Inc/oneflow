#include "oneflow/core/kernel/reduce_gather_kernel.h"

namespace oneflow {

void ReduceGatherKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& input_bns = this->op_attribute().input_bns();
  Blob* out_blob = BnInOp2Blob("out");
  char* dst_dptr = out_blob->mut_dptr<char>();
  const PbRf<int64_t>& offset = kernel_conf().reduce_gather_conf().offset();
  FOR_RANGE(int32_t, i, 0, input_bns.size()) {
    const std::string& ibn = input_bns.Get(i);
    Blob* in_blob = BnInOp2Blob(ibn);
    if (in_blob == nullptr) { continue; }
    size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
    AutoMemcpy(ctx.device_ctx, dst_dptr + offset.Get(i), in_blob->dptr<char>(), in_byte_size,
               in_blob->mem_case(), out_blob->mem_case());
  }
}

REGISTER_KERNEL(OperatorConf::kReduceGatherConf, ReduceGatherKernel);

}  // namespace oneflow
