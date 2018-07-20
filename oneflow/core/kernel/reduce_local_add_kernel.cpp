#include "oneflow/core/kernel/reduce_local_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceLocalAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto* other_val = static_cast<std::pair<int64_t, bool>*>(ctx.other);
  int64_t in_bn_id = other_val->first;
  bool is_first = other_val->second;

  const PbRpf<std::string>& output_bns = this->op_attribute().output_bns();
  Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
  size_t byte_offset = 0;
  int64_t elem_offset = 0;
  FOR_RANGE(int32_t, i, 0, output_bns.size()) {
    Blob* out_blob = BnInOp2Blob(output_bns.Get(i));
    size_t out_byte_size = out_blob->ByteSizeOfDataContentField();
    int64_t out_elem_cnt = out_blob->shape().elem_cnt();

    if (is_first) {
      Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(),
                          in_blob->dptr<char>() + byte_offset, out_byte_size);
    } else {
      KernelUtil<device_type, T>::Axpy(ctx.device_ctx, out_elem_cnt, 1.0,
                                       in_blob->dptr<T>() + elem_offset, 1, out_blob->mut_dptr<T>(),
                                       1);
    }
    byte_offset += out_byte_size;
    elem_offset += out_elem_cnt;
  }
  CHECK_EQ(in_blob->ByteSizeOfDataContentField(), byte_offset);
  CHECK_EQ(in_blob->shape().elem_cnt(), elem_offset);
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceLocalAddConf, ReduceLocalAddKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
