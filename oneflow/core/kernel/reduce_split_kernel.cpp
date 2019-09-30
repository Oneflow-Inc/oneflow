#include "oneflow/core/kernel/reduce_split_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReduceSplitKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const char* src_cur_dptr = in_blob->dptr<char>();
  for (const std::string& obn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);
    size_t out_byte_size = out_blob->AlignedByteSizeOfBlobBody();
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), src_cur_dptr, out_byte_size);
    src_cur_dptr += out_byte_size;
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReduceSplitConf, ReduceSplitKernel);

}  // namespace oneflow
