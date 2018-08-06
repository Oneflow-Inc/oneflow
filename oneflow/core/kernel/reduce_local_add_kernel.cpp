#include "oneflow/core/kernel/reduce_local_add_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceLocalAddKernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto* other_val = static_cast<std::tuple<int64_t, int64_t, bool>*>(ctx.other);
  int32_t in_bn_id = std::get<0>(*other_val);
  int32_t out_bn_id = std::get<1>(*other_val);
  bool is_inited = std::get<2>(*other_val);

  Blob* in_blob = BnInOp2Blob(this->op_attribute().input_bns().Get(in_bn_id));
  Blob* out_blob = BnInOp2Blob(this->op_attribute().output_bns().Get(out_bn_id));
  if (is_inited == false) {
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                        out_blob->ByteSizeOfDataContentField());
  } else {
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, out_blob->shape().elem_cnt(), 1.0,
                                     in_blob->dptr<T>(), 1, out_blob->mut_dptr<T>(), 1);
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceLocalAddConf, ReduceLocalAddKernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
