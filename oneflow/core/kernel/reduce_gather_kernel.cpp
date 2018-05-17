#include "oneflow/core/kernel/reduce_gather_kernel.h"

namespace oneflow {

void ReduceGatherKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const PbRpf<std::string>& input_bns = this->op_attribute().input_bns();
  Blob* out_blob = BnInOp2Blob("out");
  char* dst_dptr = out_blob->mut_dptr<char>();
  int64_t processed_regst_cnt = reinterpret_cast<int64_t>(ctx.other);
  int64_t piece_id = processed_regst_cnt / input_bns.size();
  const PbRf<int64_t>& offset = kernel_conf().reduce_gather_conf().offset();
  FOR_RANGE(int32_t, i, 0, input_bns.size()) {
    Blob* in_blob = BnInOp2Blob(input_bns.Get(i));
    if (in_blob == nullptr || in_blob->piece_id() != piece_id) { continue; }
    size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
    AutoMemcpy(ctx.device_ctx, dst_dptr + offset.Get(i), in_blob->dptr<char>(), in_byte_size,
               in_blob->mem_case(), out_blob->mem_case());
  }
}

REGISTER_KERNEL(OperatorConf::kReduceGatherConf, ReduceGatherKernel);

}  // namespace oneflow
