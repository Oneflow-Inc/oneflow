#include "oneflow/core/kernel/reduce_split_kernel.h"

namespace oneflow {

template<DeviceType device_type>
void ReduceSplitKernel<device_type>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const Blob* in_blob = BnInOp2Blob("in");
  const ReduceSplitOpConf& reduce_split_conf = this->op_conf().reduce_split_conf();
  CHECK_EQ(reduce_split_conf.out_size(), this->op_attribute().output_bns().size());
  for (int32_t out_bn_id = 0; out_bn_id < reduce_split_conf.out_size(); ++out_bn_id) {
    const char* src_cur_dptr = in_blob->dptr<char>() + reduce_split_conf.data_offset(out_bn_id);
    Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns().Get(out_bn_id));
    size_t out_byte_size = out_blob->ByteSizeOfBlobBody();
    CHECK(src_cur_dptr + out_byte_size <= in_blob->dptr<char>() + in_blob->ByteSizeOfBlobBody());
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), src_cur_dptr, out_byte_size);
  }
}

ADD_DEVICE_TYPE_KERNEL_CREATOR(OperatorConf::kReduceSplitConf, ReduceSplitKernel);

}  // namespace oneflow
