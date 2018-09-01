#include "oneflow/core/kernel/reduce_scatter_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReduceScatterKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  bool is_inplace = *static_cast<bool*>(ctx.other);
  const Blob* in_blob = BnInOp2Blob("in");
  const char* src_cur_dptr = in_blob->dptr<char>();
  for (const std::string& obn : this->op_attribute().output_bns()) {
    Blob* out_blob = BnInOp2Blob(obn);
    size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
    if (is_inplace) { CHECK_EQ(out_blob->mut_dptr<char>(), src_cur_dptr); }
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), src_cur_dptr, out_byte_size);
    src_cur_dptr += out_byte_size;
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReduceScatterConf, ReduceScatterKernel);

}  // namespace oneflow
