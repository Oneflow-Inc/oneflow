#include "oneflow/core/kernel/reduce_concat_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReduceConcatKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto* other_val = static_cast<std::pair<int64_t, bool>*>(ctx.other);
  int64_t in_bn_id = other_val->first;
  bool is_inplace = other_val->second;
  if (is_inplace) { return; }
  Blob* out_blob = BnInOp2Blob("out");
  char* dst_cur_dptr = out_blob->mut_dptr<char>();
  dst_cur_dptr += this->kernel_conf().reduce_concat_conf().data_offset().Get(in_bn_id);
  Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
  size_t in_byte_size = in_blob->ByteSizeOfDataContentField();
  Memcpy<device_type>(ctx.device_ctx, dst_cur_dptr, in_blob->dptr<char>(), in_byte_size);
}

template<DeviceType device_type>
void ReduceConcatKernel<device_type>::ForwardPackedHeader(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto* other_val = static_cast<std::pair<int64_t, bool>*>(ctx.other);
  int64_t in_bn_id = other_val->first;
  bool is_inplace = other_val->second;
  if (is_inplace) { return; }
  Blob* out_blob = BnInOp2Blob("out");
  Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
  Memcpy<device_type>(ctx.device_ctx, out_blob->mut_header_ptr(), in_blob->header_ptr(),
                      out_blob->ByteSizeOfBlobHeader());
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReduceConcatConf, ReduceConcatKernel);

}  // namespace oneflow
