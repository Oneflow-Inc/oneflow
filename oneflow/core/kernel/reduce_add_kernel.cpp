#include "oneflow/core/kernel/reduce_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto* other_val = static_cast<std::tuple<int64_t, bool, bool>*>(ctx.other);
  int64_t in_bn_id = std::get<0>(*other_val);
  bool is_out_blob_inited = std::get<1>(*other_val);
  bool is_inplace_in_blob = std::get<2>(*other_val);

  if (is_inplace_in_blob) { return; }

  Blob* out_blob = BnInOp2Blob("out");
  Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
  if (is_out_blob_inited) {
    int64_t elem_cnt = out_blob->shape().elem_cnt();
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, in_blob->dptr<T>(), 1,
                                     out_blob->mut_dptr<T>(), 1);
  } else {
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                        out_blob->ByteSizeOfBlobBody());
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceAddConf, ReduceAddKernel, FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
