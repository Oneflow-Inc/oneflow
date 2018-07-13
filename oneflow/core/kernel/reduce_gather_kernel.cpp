#include "oneflow/core/kernel/reduce_gather_kernel.h"

namespace oneflow {

void ReduceGatherKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const std::string& input_bn = *static_cast<std::string*>(ctx.other);

  Blob* out_blob = BnInOp2Blob("out");
  char* dst_cur_dptr = out_blob->mut_dptr<char>();
  int64_t i = 0;
  for (const std::string& ibn : this->op_attribute().input_bns()) {
    if (ibn == input_bn) {
      dst_cur_dptr += this->kernel_conf().reduce_gather_conf().data_offset().Get(i);
      Blob* in_blob = BnInOp2Blob(ibn);
      CHECK(in_blob != nullptr);
      size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
      Memcpy<DeviceType::kGPU>(ctx.device_ctx, dst_cur_dptr, in_blob->dptr<char>(), in_byte_size);
    }
    i += 1;
  }
}

REGISTER_KERNEL(OperatorConf::kReduceGatherConf, ReduceGatherKernel);

}  // namespace oneflow
