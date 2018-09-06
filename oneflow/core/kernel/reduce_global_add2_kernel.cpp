#include "oneflow/core/kernel/reduce_global_add2_kernel.h"

namespace oneflow {

template<DeviceType device_type, typename T>
void ReduceGlobalAdd2Kernel<device_type, T>::ForwardDataContent(
    const KernelCtx& ctx, std::function<Blob*(const std::string&)> BnInOp2Blob) const {
  const auto* other_val = static_cast<std::tuple<std::string, bool, bool>*>(ctx.other);
  const std::string& ibn = std::get<0>(*other_val);
  bool is_out_blob_inited = std::get<1>(*other_val);
  bool is_inplace_in_blob = std::get<2>(*other_val);

  if (is_inplace_in_blob) { return; }

  Blob* out_blob = BnInOp2Blob("out");
  Blob* in_blob = BnInOp2Blob(ibn);
  if (is_out_blob_inited) {
    int64_t elem_cnt = out_blob->shape().elem_cnt();
    KernelUtil<device_type, T>::Axpy(ctx.device_ctx, elem_cnt, 1.0, in_blob->dptr<T>(), 1,
                                     out_blob->mut_dptr<T>(), 1);
  } else {
    Memcpy<device_type>(ctx.device_ctx, out_blob->mut_dptr<char>(), in_blob->dptr<char>(),
                        out_blob->ByteSizeOfDataContentField());
  }
}

ADD_DEFAULT_KERNEL_CREATOR(OperatorConf::kReduceGlobalAdd2Conf, ReduceGlobalAdd2Kernel,
                           FLOATING_DATA_TYPE_SEQ);

}  // namespace oneflow
