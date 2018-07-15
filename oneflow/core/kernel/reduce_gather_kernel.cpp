#include "oneflow/core/kernel/reduce_gather_kernel.h"

namespace oneflow {

void ReduceGatherKernel::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  int64_t in_bn_id = *static_cast<int64_t*>(ctx.other);

  Blob* out_blob = BnInOp2Blob("out");
  char* dst_cur_dptr = out_blob->mut_dptr<char>();
  dst_cur_dptr += this->kernel_conf().reduce_gather_conf().data_offset().Get(in_bn_id);
  Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
  size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  Memcpy<DeviceType::kGPU>(ctx.device_ctx, dst_cur_dptr, in_blob->dptr<char>(), in_byte_size);
}

REGISTER_KERNEL(OperatorConf::kReduceGatherConf, ReduceGatherKernel);

}  // namespace oneflow
